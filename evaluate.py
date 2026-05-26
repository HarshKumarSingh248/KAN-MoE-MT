"""
evaluate.py — Evaluate KAN-MoE-MT on test & challenge sets.

Usage:
    python evaluate.py \
        --lang hindi \
        --ckpt /path/to/best_model.pt \
        --data-dir /path/to/hindi-visual-genome-11 \
        --nllb-dir /path/to/nllb13b_local \
        --ribes /path/to/RIBES.py \
        --indic-nlp /path/to/indic_nlp_library \
        --splits test challenge
"""

import os
import sys

# Fix GLIBCXX_3.4.29 missing on older systems (if needed).
# Users can manually set LD_LIBRARY_PATH before running if required.
_HERE = os.path.dirname(os.path.abspath(__file__))

import argparse
import json
import logging
import os
import sys

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import transformers
transformers.logging.set_verbosity_error()

from config import get_cfg
from data_utils import load_split, compute_bleu, compute_ribes, HVGDataset, make_collate_fn
from model import KANMoENLLB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@torch.no_grad()
def generate_all(model, tokenizer, df, cfg, device):
    model.eval()
    dataset = HVGDataset(df)
    collate = make_collate_fn(
        tokenizer, cfg.src_lang_nllb, cfg.tgt_lang_nllb,
        cfg.max_src_len, cfg.max_tgt_len,
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size * 2,
                        collate_fn=collate, num_workers=2, pin_memory=True)
    hyps, refs = [], []
    for batch in loader:
        preds = model.generate(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["bbox"].to(device),
            tokenizer,
        )
        hyps.extend(preds)
        refs.extend(batch["tgt_texts"])
    return hyps, refs


HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang",      default="hindi", choices=["hindi", "bengali"])
    parser.add_argument("--ckpt",      default=os.path.join(HERE, "best_model.pt"))
    parser.add_argument("--data-dir",  required=True, help="Path to visual genome dataset dir")
    parser.add_argument("--nllb-dir",  default=os.path.join(HERE, "nllb13b_local"))
    parser.add_argument("--ribes",     default=os.path.join(HERE, "RIBES.py"))
    parser.add_argument("--moses",     default=os.path.join(HERE, "mosesdecoder/scripts/generic/multi-bleu.perl"))
    parser.add_argument("--indic-nlp", default=os.path.join(HERE, "indic_nlp_library"))
    parser.add_argument("--splits",    nargs="+", default=["test", "challenge"],
                        choices=["dev", "test", "challenge"])
    parser.add_argument("--out-dir",   default=HERE, help="Where to save eval_results.json")
    args = parser.parse_args()

    cfg = get_cfg(
        lang          = args.lang,
        data_dir      = args.data_dir,
        nllb13b_local = args.nllb_dir,
        work_dir      = args.out_dir,
        ribes_script  = args.ribes,
        moses_script  = args.moses,
        indic_nlp_dir = args.indic_nlp,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.ckpt):
        logger.error(f"Checkpoint not found: {args.ckpt}")
        sys.exit(1)

    logger.info("=" * 65)
    logger.info(f"KAN-MoE-MT Evaluation  —  {args.lang.upper()}")
    logger.info(f"Checkpoint : {args.ckpt}")
    logger.info(f"Device     : {device}")
    logger.info(f"Splits     : {args.splits}")
    logger.info("=" * 65)

    logger.info("Loading model ...")
    model = KANMoENLLB(cfg).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters — total: {total:,}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.nllb13b_local, local_files_only=True)
    tokenizer.src_lang = cfg.src_lang_nllb
    tokenizer.tgt_lang = cfg.tgt_lang_nllb

    results = {}
    for split in args.splits:
        logger.info(f"\nEvaluating {split.upper()} ...")
        df = load_split(cfg, split)
        hyps, refs = generate_all(model, tokenizer, df, cfg, device)
        bleu  = compute_bleu(hyps, refs, cfg)
        ribes = compute_ribes(hyps, refs, cfg)
        results[split] = {"BLEU": round(bleu, 2), "RIBES": round(ribes, 4), "n": len(refs)}
        logger.info(f"  BLEU  = {bleu:.2f}")
        logger.info(f"  RIBES = {ribes:.4f}")
        logger.info(f"  n     = {len(refs):,}")

    logger.info(f"\n{'='*55}")
    logger.info(f"{'Split':<12} {'BLEU':>8} {'RIBES':>8} {'n':>6}")
    logger.info(f"{'-'*55}")
    for split, r in results.items():
        logger.info(f"{split:<12} {r['BLEU']:>8.2f} {r['RIBES']:>8.4f} {r['n']:>6,}")
    logger.info(f"{'='*55}")

    out_file = os.path.join(args.out_dir, "eval_results.json")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"\nResults saved → {out_file}")


if __name__ == "__main__":
    main()
