
import os
import sys
import re
import logging
import subprocess
import tempfile
from collections import Counter
from PIL import Image
import torch
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import Dataset, DataLoader, DistributedSampler

logger = logging.getLogger(__name__)

_COLS = ["image_id", "x", "y", "w", "h", "en", "tgt"]

# ── TSV loading ────────────────────────────────────────────────────────────────

def read_tsv(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path, sep="\t", header=None, names=_COLS,
        dtype={"image_id": str}, quoting=3,
        keep_default_na=False, encoding="utf-8",
    )
    for c in ["x", "y", "w", "h"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def _normalize_bbox(df: pd.DataFrame, img_ref_w: int, img_ref_h: int) -> pd.DataFrame:
    df = df.copy()
    df["x"] = (df["x"] / img_ref_w).clip(0.0, 1.0)
    df["w"] = (df["w"] / img_ref_w).clip(0.0, 1.0)
    df["y"] = (df["y"] / img_ref_h).clip(0.0, 1.0)
    df["h"] = (df["h"] / img_ref_h).clip(0.0, 1.0)
    return df


def load_split(cfg, split: str) -> pd.DataFrame:
    split_map = {
        "train": "train",
        "dev": "dev",
        "test": "test",
        "challenge": "challenge-test-set",
    }
    name = split_map[split]
    tsv = os.path.join(cfg.data_dir, f"{cfg.tsv_prefix}-{name}.txt")
    df = read_tsv(tsv)
    df = _normalize_bbox(df, cfg.img_ref_w, cfg.img_ref_h)
    logger.info(f"Loaded {split} ({len(df):,} samples) from {tsv}")
    return df


def _img_dir_for_split(cfg, split: str) -> str:
    # Adjust path according to your actual image directory
    return os.path.join(cfg.data_dir, "images")


# ── Text-only Dataset ─────────────────────────────────────────────────────────

class HVGDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.image_ids = df["image_id"].tolist()
        self.en   = df["en"].tolist()
        self.tgt  = df["tgt"].tolist()
        self.bbox = torch.tensor(
            df[["x", "y", "w", "h"]].values.astype("float32"),
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.en)

    def __getitem__(self, idx):
        return {
            "image_id": self.image_ids[idx],
            "en":       self.en[idx],
            "tgt":      self.tgt[idx],
            "bbox":     self.bbox[idx],
        }


# ── Visual Dataset ────────────────────────────────────────────────────────────

class HVGVisualDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str):
        self.df = df
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['image_id']}.jpg")
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except Exception:
            image = torch.zeros(3, 224, 224)
        return {
            "en":   row["en"],
            "tgt":  row["tgt"],
            "bbox": torch.tensor([row["x"], row["y"], row["w"], row["h"]], dtype=torch.float32),
            "image": image,
        }


# ── Collate functions ─────────────────────────────────────────────────────────

def make_collate_fn(tokenizer, src_lang: str, tgt_lang: str,
                    max_src_len: int = 80, max_tgt_len: int = 80):
    def collate(batch):
        image_ids = [b["image_id"] for b in batch]
        srcs      = [b["en"]       for b in batch]
        tgts      = [b["tgt"]      for b in batch]
        bboxes    = torch.stack([b["bbox"] for b in batch])

        # Encode source
        tokenizer.src_lang = src_lang
        enc = tokenizer(
            srcs, return_tensors="pt", padding=True,
            truncation=True, max_length=max_src_len,
        )

        # Encode target: use src_lang so no language token is prepended,
        # then manually prepend forced_bos_token_id as first label token.
        # This gives: [hin_Deva, tok1, tok2, ..., </s>] which is what
        # NLLB decoder expects during teacher-forced training.
        tokenizer.src_lang = src_lang
        dec = tokenizer(
            tgts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_tgt_len - 1,  # -1 to leave room for bos
        )
        tokenizer.src_lang = src_lang

        forced_bos = tokenizer.convert_tokens_to_ids(tgt_lang)
        bos_col = torch.full((dec["input_ids"].size(0), 1), forced_bos, dtype=torch.long)
        labels = torch.cat([bos_col, dec["input_ids"]], dim=1)
        labels[labels == tokenizer.pad_token_id] = -100        # pad=1
        labels[labels == tokenizer.unk_token_id] = -100        # unk=3 (appended by fast tokenizer)

        return {
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels":         labels,
            "bbox":           bboxes,
            "src_texts":      srcs,
            "tgt_texts":      tgts,
            "image_ids":      image_ids,
        }
    return collate


def make_visual_collate_fn(tokenizer, src_lang: str, tgt_lang: str,
                           max_src_len: int = 80, max_tgt_len: int = 80):
    def collate(batch):
        srcs   = [b["en"]    for b in batch]
        tgts   = [b["tgt"]   for b in batch]
        bboxes = torch.stack([b["bbox"] for b in batch])
        images = torch.stack([b["image"] for b in batch])

        tokenizer.src_lang = src_lang
        enc = tokenizer(
            srcs, return_tensors="pt", padding=True,
            truncation=True, max_length=max_src_len,
        )

        tokenizer.src_lang = tgt_lang
        dec = tokenizer(
            tgts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_tgt_len,
        )
        tokenizer.src_lang = src_lang

        labels = dec["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100

        return {
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels":         labels,
            "bbox":           bboxes,
            "images":         images,
            "src_texts":      srcs,
            "tgt_texts":      tgts,
        }
    return collate


def make_loader(df: pd.DataFrame, tokenizer, cfg, split: str, distributed: bool = False):
    dataset = HVGDataset(df)
    collate = make_collate_fn(
        tokenizer, cfg.src_lang_mbart, cfg.tgt_lang_mbart,
        cfg.max_src_len, cfg.max_tgt_len,
    )
    sampler = DistributedSampler(dataset, shuffle=(split == "train")) if distributed else None
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=(split == "train" and sampler is None),
        sampler=sampler,
        collate_fn=collate,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=(split == "train"),
    )


def make_visual_loader(df: pd.DataFrame, tokenizer, cfg, split: str, distributed: bool = False):
    img_dir = _img_dir_for_split(cfg, split)
    dataset = HVGVisualDataset(df, img_dir)
    collate = make_visual_collate_fn(
        tokenizer, cfg.src_lang_mbart, cfg.tgt_lang_mbart,
        cfg.max_src_len, cfg.max_tgt_len,
    )
    sampler = DistributedSampler(dataset, shuffle=(split == "train")) if distributed else None
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=(split == "train" and sampler is None),
        sampler=sampler,
        collate_fn=collate,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=(split == "train"),
    )


# ── WAT tokenisation (indic_nlp) ──────────────────────────────────────────────

_normalizers: dict = {}
_trivial_tokenize = None
_indic_ready: set = set()


def _init_indic(indic_nlp_dir: str, lang_code: str) -> bool:
    global _trivial_tokenize
    if lang_code in _indic_ready:
        return True
    for p in [indic_nlp_dir, os.path.join(indic_nlp_dir, "src")]:
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        from indicnlp import common as indic_common
        res_path = os.path.join(indic_nlp_dir, "indicnlp_resources")
        if os.path.isdir(res_path):
            indic_common.set_resources_path(res_path)
        from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
        from indicnlp.tokenize import indic_tokenize as _itok
        _normalizers[lang_code] = IndicNormalizerFactory().get_normalizer(lang_code)
        _trivial_tokenize = _itok.trivial_tokenize
        _indic_ready.add(lang_code)
        return True
    except Exception as e:
        logger.warning(f"indic_nlp unavailable ({e}). Falling back to whitespace tokenisation.")
        _normalizers[lang_code] = None
        _indic_ready.add(lang_code)
        return False


def wat_tok(text: str, lang_code: str, indic_nlp_dir: str) -> str:
    _init_indic(indic_nlp_dir, lang_code)
    norm = _normalizers.get(lang_code)
    if norm is not None:
        text = norm.normalize(text)
    if _trivial_tokenize is not None:
        return " ".join(_trivial_tokenize(text, lang_code))
    return " ".join(text.split())


# ── BLEU via Moses multi-bleu.perl (WAT evaluation protocol) ─────────────────

_MOSES_MULTI_BLEU = "/home/abhishara_iitp/mosesdecoder-RELEASE-2.1.1/scripts/generic/multi-bleu.perl"


def compute_bleu(hyps: list[str], refs: list[str], cfg) -> float:
    ht = [wat_tok(h, cfg.lang_code, cfg.indic_nlp_dir) for h in hyps]
    rt = [wat_tok(r, cfg.lang_code, cfg.indic_nlp_dir) for r in refs]

    with tempfile.TemporaryDirectory() as tmp:
        hyp_f = os.path.join(tmp, "hyp.txt")
        ref_f = os.path.join(tmp, "ref.txt")
        with open(hyp_f, "w", encoding="utf-8") as f:
            f.write("\n".join(ht) + "\n")
        with open(ref_f, "w", encoding="utf-8") as f:
            f.write("\n".join(rt) + "\n")
        try:
            # multi-bleu.perl -lc matches the official WAT evaluation protocol
            with open(hyp_f) as hyp_in:
                result = subprocess.run(
                    ["perl", _MOSES_MULTI_BLEU, "-lc", ref_f],
                    stdin=hyp_in, capture_output=True, text=True, timeout=180,
                )
            # Output format: BLEU = 12.34, ...
            m = re.search(r"BLEU\s*=\s*([\d.]+)", result.stdout)
            if m:
                return float(m.group(1))
            logger.warning(f"multi-bleu.perl unexpected output: {result.stdout.strip()}")
        except Exception as e:
            logger.warning(f"multi-bleu.perl call failed: {e}")
    return 0.0


# ── RIBES via subprocess ──────────────────────────────────────────────────────

def compute_ribes(hyps: list[str], refs: list[str], cfg) -> float:
    ht = [wat_tok(h, cfg.lang_code, cfg.indic_nlp_dir) for h in hyps]
    rt = [wat_tok(r, cfg.lang_code, cfg.indic_nlp_dir) for r in refs]

    with tempfile.TemporaryDirectory() as tmp:
        hyp_f = os.path.join(tmp, "hyp.txt")
        ref_f = os.path.join(tmp, "ref.txt")
        with open(hyp_f, "w", encoding="utf-8") as f:
            f.write("\n".join(ht) + "\n")
        with open(ref_f, "w", encoding="utf-8") as f:
            f.write("\n".join(rt) + "\n")
        try:
            result = subprocess.run(
                [sys.executable, cfg.ribes_script, "-r", ref_f, hyp_f],
                capture_output=True, text=True, timeout=180,
            )
            for line in result.stdout.strip().splitlines():
                line = line.strip()
                if re.match(r"^[\d.]+", line):
                    return float(line.split()[0])
        except Exception as e:
            logger.warning(f"RIBES.py call failed: {e}")
    return 0.0