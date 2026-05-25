"""
train.py — Train KAN-MoE-MT (NLLB-1.3B + KAN-MoE + RegionGate) on Hindi/Bengali Visual Genome.

Full finetuning — no freezing. All 2.4B params train from epoch 1 with discriminative LRs:
  KAN/RegionGate: 3e-4  |  decoder: 1e-5  |  encoder: 1e-5

Single GPU:
    python train.py --lang hindi

Multi-GPU:
    torchrun --nproc_per_node=2 train.py --lang hindi
    torchrun --nproc_per_node=2 train.py --lang bengali

Best model saved to: {work_dir}/best_model.pt
"""

# ── GLIBCXX fix: re-exec with correct LD_LIBRARY_PATH before any native libs load ──
import os, sys, warnings
_LIBSTDCXX_CANDIDATES = [
    "/home/abhishara_iitp/.conda/envs/bclm/lib",
    "/home/abhishara_iitp/.conda/envs/vg/lib",
]
if not os.environ.get("_GLIBCXX_FIXED"):
    for _lib in _LIBSTDCXX_CANDIDATES:
        if os.path.isdir(_lib):
            os.environ["LD_LIBRARY_PATH"] = _lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
            os.environ["_GLIBCXX_FIXED"] = "1"
            os.execv(sys.executable, [sys.executable] + sys.argv)

warnings.filterwarnings("ignore", message=".*tie.*word.*embeddings.*")
warnings.filterwarnings("ignore", message=".*tied weights.*")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import transformers
transformers.logging.set_verbosity_error()

from config import get_cfg, CFG
from data_utils import (
    load_split, HVGDataset, make_collate_fn,
    compute_bleu, compute_ribes,
)
from model import KANMoENLLB


# ── Paths (edit these if your server layout differs) ──────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))

_DATA_DIR    = os.path.join(_HERE, "data")
_NLLB13B_DIR = os.path.join(_HERE, "nllb13b_local")
_RIBES       = os.path.join(_HERE, "RIBES.py")
_INDIC_NLP   = os.path.join(_HERE, "indic_nlp_library")

_WORK_DIR_HINDI   = os.path.join(_HERE, "runs", "hindi")
_WORK_DIR_BENGALI = os.path.join(_HERE, "runs", "bengali")

_BENGALI_DATA_DIR = os.path.join(_HERE, "data")


# ── Distributed helpers ────────────────────────────────────────────────────────

def setup_ddp():
    rank       = int(os.environ.get("RANK",       0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        try:
            dist.init_process_group(backend="nccl")
        except Exception:
            dist.init_process_group(backend="gloo")
        torch.cuda.set_device(local_rank)
    return local_rank, world_size, rank


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── DataLoader ────────────────────────────────────────────────────────────────

def make_nllb_loader(df, tokenizer, cfg: CFG, split: str, distributed: bool = False):
    dataset = HVGDataset(df)
    collate = make_collate_fn(
        tokenizer,
        src_lang=cfg.src_lang_nllb,
        tgt_lang=cfg.tgt_lang_nllb,
        max_src_len=cfg.max_src_len,
        max_tgt_len=cfg.max_tgt_len,
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


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_all(model: KANMoENLLB, tokenizer, df, cfg: CFG, device):
    model.eval()
    dataset = HVGDataset(df)
    collate = make_collate_fn(
        tokenizer,
        src_lang=cfg.src_lang_nllb,
        tgt_lang=cfg.tgt_lang_nllb,
        max_src_len=cfg.max_src_len,
        max_tgt_len=cfg.max_tgt_len,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size * 2,
        collate_fn=collate,
        num_workers=2,
        pin_memory=True,
    )
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


# ── Main training loop ────────────────────────────────────────────────────────

def run(cfg: CFG, resume_from: int = 0, resume_ckpt: str = None, best_bleu_override: float = -1.0):
    local_rank, world_size, rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed + rank)

    if is_main(rank):
        Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
        log_file = os.path.join(cfg.work_dir, "train.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)s  %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_file, mode="w"),
            ],
        )
    else:
        logging.basicConfig(level=logging.WARNING)

    logger = logging.getLogger(__name__)

    if is_main(rank):
        logger.info("=" * 65)
        logger.info("KAN-MoE-MT Training")
        logger.info(f"  Language  : {cfg.lang}")
        logger.info(f"  Work dir  : {cfg.work_dir}")
        logger.info(f"  World size: {world_size}")
        logger.info(f"  Device    : {device}")
        logger.info(f"  Backbone  : NLLB-200-1.3B  (full finetune, no freezing)")
        logger.info(f"  KAN-MoE   : {cfg.n_experts} experts, hidden={cfg.kan_hidden}, basis={cfg.kan_basis}")
        logger.info(f"  RegionGate: additive Fourier bias (before KAN-MoE)")
        logger.info(f"  LR        : kan={cfg.lr_kan}  decoder={cfg.lr_decoder}  encoder={cfg.lr_encoder}")
        logger.info(f"  Smoothing : 0.1 | beam={cfg.beam_size} | no_repeat={cfg.no_repeat_ngram}")
        logger.info("=" * 65)

    # ── Tokenizer ─────────────────────────────────────────────────────────
    # Load ONLY from local files (no internet access on this server)
    if not os.path.isdir(cfg.nllb13b_local):
        if is_main(rank):
            logger.error(f"ERROR: NLLB directory not found: {cfg.nllb13b_local}")
            logger.error("Server has no internet access. Please ensure nllb13b_local exists.")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(cfg.nllb13b_local, local_files_only=True)
    tokenizer.src_lang = cfg.src_lang_nllb
    tokenizer.tgt_lang = cfg.tgt_lang_nllb

    # ── Data ──────────────────────────────────────────────────────────────
    train_df = load_split(cfg, "train")
    dev_df   = load_split(cfg, "dev")

    train_loader = make_nllb_loader(train_df, tokenizer, cfg, "train",
                                    distributed=(world_size > 1))
    dev_loader   = make_nllb_loader(dev_df, tokenizer, cfg, "dev", distributed=False)

    if is_main(rank):
        logger.info(f"Train batches: {len(train_loader)}  Dev batches: {len(dev_loader)}")

    # ── Model (full finetune — no freezing) ───────────────────────────────
    model = KANMoENLLB(cfg).to(device)

    if resume_ckpt and os.path.exists(resume_ckpt):
        model.load_state_dict(torch.load(resume_ckpt, map_location=device))
        if is_main(rank):
            logger.info(f"Loaded checkpoint: {resume_ckpt}")

    if is_main(rank):
        counts = model.param_counts()
        logger.info(f"Params — total: {counts['total']:,}  trainable: {counts['trainable']:,}")

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    raw: KANMoENLLB = model.module if hasattr(model, "module") else model

    # ── Optimizer & scheduler ─────────────────────────────────────────────
    optimizer   = torch.optim.AdamW(
        raw.get_param_groups(), weight_decay=0.01, betas=(0.9, 0.98), eps=1e-6,
    )
    total_steps = (len(train_loader) // cfg.grad_accum) * cfg.epochs
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda")

    best_bleu        = best_bleu_override if best_bleu_override > 0 else -1.0
    patience_counter = 0
    history          = []
    ckpt_path        = os.path.join(cfg.work_dir, "best_model.pt")

    if is_main(rank) and resume_from:
        logger.info(f"Resuming from epoch {resume_from + 1}, best_bleu={best_bleu:.2f}")

    # ── Training loop ─────────────────────────────────────────────────────
    epoch_bar = tqdm(
        range(resume_from + 1, cfg.epochs + 1),
        desc="Overall", disable=not is_main(rank),
        ncols=100, unit="ep", file=sys.stdout,
    )

    for epoch in epoch_bar:
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        aux_accum  = 0.0
        optimizer.zero_grad()
        n_batches  = len(train_loader)

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:2d}/{cfg.epochs}",
            disable=not is_main(rank),
            ncols=100, file=sys.stdout, dynamic_ncols=False,
        )

        for step, batch in enumerate(pbar, 1):
            inp  = batch["input_ids"].to(device, non_blocking=True)
            msk  = batch["attention_mask"].to(device, non_blocking=True)
            lbl  = batch["labels"].to(device, non_blocking=True)
            bbox = batch["bbox"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss, aux_loss = model(
                    input_ids=inp, attention_mask=msk,
                    labels=lbl, bbox=bbox, label_smoothing=0.1,
                )

            scaler.scale(loss / cfg.grad_accum).backward()
            epoch_loss += loss.item()
            aux_accum  += aux_loss.item()

            if step % cfg.grad_accum == 0 or step == n_batches:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            if is_main(rank):
                pbar.set_postfix(
                    loss=f"{epoch_loss/step:.4f}",
                    aux=f"{aux_accum/step:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )

        pbar.close()
        avg_loss = epoch_loss / n_batches
        avg_aux  = aux_accum  / n_batches

        # ── Evaluation (rank 0 only) ───────────────────────────────────────
        if epoch % cfg.eval_every == 0 and is_main(rank):
            hyps, refs = generate_all(raw, tokenizer, dev_df, cfg, device)
            bleu  = compute_bleu(hyps, refs, cfg)
            ribes = compute_ribes(hyps, refs, cfg)

            logger.info(
                f"Epoch {epoch:3d}/{cfg.epochs} | "
                f"loss={avg_loss:.4f}  aux={avg_aux:.4f} | "
                f"Dev BLEU={bleu:.2f}  RIBES={ribes:.4f}"
            )
            history.append({"epoch": epoch, "loss": avg_loss, "bleu": bleu, "ribes": ribes})

            if bleu > best_bleu:
                best_bleu        = bleu
                patience_counter = 0
                torch.save(raw.state_dict(), ckpt_path)
                logger.info(f"  → Saved best_model.pt  (BLEU={bleu:.2f})")
            else:
                patience_counter += 1
                logger.info(f"  No improvement ({patience_counter}/{cfg.patience})")
                if patience_counter >= cfg.patience:
                    logger.info("Early stopping triggered.")

            epoch_bar.set_postfix(
                BLEU=f"{bleu:.2f}", best=f"{best_bleu:.2f}",
                patience=f"{patience_counter}/{cfg.patience}",
            )

        # Sync early-stop flag across ranks
        if world_size > 1:
            stop_flag = torch.tensor(
                int(patience_counter >= cfg.patience), device=device
            )
            dist.broadcast(stop_flag, src=0)
            if stop_flag.item():
                break

    epoch_bar.close()

    # ── Final evaluation on test + challenge ──────────────────────────────
    if is_main(rank):
        logger.info("Loading best_model.pt for final evaluation ...")
        raw.load_state_dict(torch.load(ckpt_path, map_location=device))

        results = {
            "lang":          cfg.lang,
            "best_dev_bleu": best_bleu,
            "history":       history,
        }

        for split in ["test", "challenge"]:
            split_df = load_split(cfg, split)
            hyps, refs = generate_all(raw, tokenizer, split_df, cfg, device)
            bleu  = compute_bleu(hyps, refs, cfg)
            ribes = compute_ribes(hyps, refs, cfg)
            results[split] = {"BLEU": round(bleu, 2), "RIBES": round(ribes, 4), "n": len(refs)}
            logger.info(f"[{split:>12}]  BLEU={bleu:.2f}  RIBES={ribes:.4f}  (n={len(refs)})")

        out_file = os.path.join(cfg.work_dir, "final_results.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved → {out_file}")

    cleanup()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train KAN-MoE-MT (NLLB-1.3B + KAN-MoE + RegionGate)."
    )
    parser.add_argument(
        "--lang", choices=["hindi", "bengali"], default="hindi",
        help="Target language (default: hindi)",
    )
    parser.add_argument("--data-dir", type=str, default=_DATA_DIR,
                        help=f"Path to dataset directory (default: {_DATA_DIR})")
    parser.add_argument("--nllb-dir", type=str, default=_NLLB13B_DIR,
                        help=f"Path to nllb13b_local directory (default: {_NLLB13B_DIR})")
    parser.add_argument("--work-dir", type=str, default=None,
                        help="Path to output directory (default: runs/{lang})")
    parser.add_argument("--indic-nlp", type=str, default=_INDIC_NLP,
                        help=f"Path to indic_nlp_library directory (default: {_INDIC_NLP})")
    parser.add_argument("--ribes", type=str, default=_RIBES,
                        help=f"Path to RIBES.py script (default: {_RIBES})")
    parser.add_argument("--resume-from", type=int, default=0,
                        help="Resume after this epoch (e.g. 7 resumes from epoch 8)")
    parser.add_argument("--resume-ckpt", type=str, default=None,
                        help="Checkpoint to load weights from when resuming")
    parser.add_argument("--best-bleu", type=float, default=-1.0,
                        help="Best BLEU achieved before resuming — prevents overwriting a better checkpoint")
    args = parser.parse_args()

    if args.lang == "hindi":
        default_work_dir = _WORK_DIR_HINDI
    else:
        default_work_dir = _WORK_DIR_BENGALI

    work_dir = args.work_dir if args.work_dir else default_work_dir

    cfg = get_cfg(
        lang=args.lang,
        data_dir=args.data_dir,
        nllb13b_local=args.nllb_dir,
        work_dir=work_dir,
        ribes_script=args.ribes,
        indic_nlp_dir=args.indic_nlp,
    )

    resume_ckpt = args.resume_ckpt
    if args.resume_from and resume_ckpt is None:
        resume_ckpt = os.path.join(work_dir, "best_model.pt")

    run(cfg, resume_from=args.resume_from, resume_ckpt=resume_ckpt, best_bleu_override=args.best_bleu)
