# DualKAN-NLLB: Visual Grounding for Hindi Machine Translation

KAN-MoE + RegionGate fusion over NLLB-200-1.3B for Hindi Visual Genome MT.

## Files

```
kan_nllb/
├── model.py        # DualKANNLLB architecture (KAN-MoE + RegionGate + NLLB-1.3B)
├── data_utils.py   # Dataset, collate, BLEU, RIBES
├── config.py       # CFG dataclass
├── evaluate.py     # Evaluation script
├── best_model.pt   # Best checkpoint (provide separately — 9.1 GB)
└── README.md
```

## Requirements

```bash
pip install torch transformers sentencepiece pandas tqdm
```

## Reproduce Results

```bash
export LD_LIBRARY_PATH=/path/to/conda/env/lib:$LD_LIBRARY_PATH

python evaluate.py \
    --lang      hindi \
    --ckpt      /path/to/best_model.pt \
    --data-dir  /path/to/hindi-visual-genome-11 \
    --nllb-dir  /path/to/nllb13b_local \
    --ribes     /path/to/RIBES.py \
    --indic-nlp /path/to/indic_nlp_library \
    --splits    test challenge \
    --out-dir   ./results
```

### Required external resources (not included, paths passed as arguments)
| Resource | Description |
|---|---|
| `best_model.pt` | Trained checkpoint (~9.1 GB) |
| `nllb13b_local/` | facebook/nllb-200-1.3B weights (HuggingFace) |
| `hindi-visual-genome-11/` | Hindi Visual Genome dataset |
| `RIBES.py` | RIBES evaluation script (WAT) |
| `indic_nlp_library/` | Indic NLP Library for WAT tokenisation |

## Expected Results (Hindi)

| Split | BLEU | RIBES |
|-------|------|-------|
| Test | ~50+ | ~0.84+ |
| Challenge | ~50+ | ~0.84+ |

SOTA (Chitranuvaad): Test=43.9, Challenge=54.7
