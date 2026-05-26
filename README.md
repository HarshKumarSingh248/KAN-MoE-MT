# KAN-MoE: Interpretable Expert Routing for Indic Machine Translation

KAN-MoE translates English image captions into Hindi/Bengali using only the **bounding box coordinates** of the described image region — no image pixels needed at inference time. It augments a fully fine-tuned NLLB-1.3B backbone with two modules:

- **RegionGate** — encodes `(x, y, w, h)` with Fourier features, injects a spatial bias into every encoder token
- **KAN-MoE** — 4 RBF-KAN experts with soft routing; learns to specialize without supervision

---

## Quick Start (TL;DR)

```bash
# 1. Clone repo and install environment
git clone https://github.com/HarshKumarSingh248/KAN-MoE-MT.git
cd KAN-MoE-MT
conda env create -f environment.yml
conda activate kan-moe-mt

# 2. Download NLLB-1.3B weights
huggingface-cli download facebook/nllb-200-1.3B --repo-type model --local-dir nllb13b_local

# 3. Install Indic NLP Library
git clone https://github.com/indicnlp/indic_nlp_library.git
cd indic_nlp_library
git clone https://github.com/indicnlp/indic_nlp_resources.git
python setup.py install
cd ..

# 4. Download dataset from WAT: https://ufal.mff.cuni.cz/wat2025english-indicmultimodaltranslation
# Extract to ./data folder

# 4.5 Download Moses decoder (for BLEU evaluation)
git clone https://github.com/moses-smt/mosesdecoder.git

# 5. Train
torchrun --nproc_per_node=2 train.py --lang hindi \
    --data-dir ./data \
    --nllb-dir ./nllb13b_local \
    --indic-nlp ./indic_nlp_library \
    --moses ./mosesdecoder/scripts/generic/multi-bleu.perl

# 6. Evaluate
python evaluate.py \
    --lang hindi \
    --ckpt ./runs/hindi/best_model.pt \
    --data-dir ./data \
    --nllb-dir ./nllb13b_local \
    --indic-nlp ./indic_nlp_library \
    --moses ./mosesdecoder/scripts/generic/multi-bleu.perl \
    --splits test challenge
```

> **Note:** All paths are customizable via command-line arguments. See [Step 5 — Train](#step-5--train) for details.

---

## Files in This Repo

```
KAN-MoE-MT/
├── model.py              — RegionGate + KAN-MoE + NLLB-1.3B architecture
├── config.py             — all hyperparameters, get_cfg()
├── data_utils.py         — TSV loading, HVGDataset, collate, BLEU, RIBES
├── train.py              — full fine-tuning, saves runs/hindi/best_model.pt
├── evaluate.py           — evaluation script
├── RIBES.py              — RIBES metric (official WAT implementation)
├── environment.yml       — conda environment (pinned versions)
├── requirements.txt      — pip alternative
└── README.md             — this file
```

**You must download separately:**

| Item | Where | Size |
|------|-------|------|
| `nllb13b_local/` | HuggingFace (Step 2) | ~5 GB |
| `data/` text files | WAT task page (Step 3) | ~4 MB |
| `data/` image folders | WAT task page (Step 3, training only) | ~4.5 GB |

There is no pre-trained checkpoint provided. **Train the model first** — it saves `runs/hindi/best_model.pt` which is then used for evaluation.

---

## Reproduce Results — Step by Step

### Step 1 — Clone and Setup Environment

```bash
git clone https://github.com/HarshKumarSingh248/KAN-MoE-MT.git
cd KAN-MoE-MT
```

**Using Conda (recommended for reproducibility):**

```bash
conda env create -f environment.yml
conda activate kan-moe-mt
```

This installs **exact pinned versions** of:
- PyTorch 2.0.0 with CUDA 12.1
- Transformers 4.36.2 (NLLB compatibility)
- All evaluation libraries (sacrebleu, scipy, scikit-learn)

**Alternative: Using pip:**

```bash
pip install -r requirements.txt
```

> Note: pip may install newer versions than tested. For strict reproducibility, use conda.

---

### Step 2 — Download NLLB-1.3B Backbone Weights (~5 GB)

```bash
huggingface-cli download facebook/nllb-200-1.3B \
    --repo-type model \
    --local-dir nllb13b_local
```

Verify download:

```bash
ls nllb13b_local/config.json nllb13b_local/tokenizer.json
# Both files should exist
```

**If HuggingFace CLI not installed:**

```bash
pip install huggingface-hub[cli]
```

**Alternative: Manual download from HuggingFace web**

1. Go to: https://huggingface.co/facebook/nllb-200-1.3B
2. Download all files to `nllb13b_local/` folder
3. Verify with command above

---

### Step 2.5 — Install Indic NLP Library (Required for Tokenization)

The **Indic NLP Library** is essential for evaluation. It provides language-specific tokenization and text normalization for Hindi/Bengali, required by the WAT evaluation protocol.

**Installation:**

```bash
git clone https://github.com/indicnlp/indic_nlp_library.git
cd indic_nlp_library
git clone https://github.com/indicnlp/indic_nlp_resources.git
python setup.py install
```

This installs:
- `indic_nlp.normalize` — normalizes Hindi/Bengali text (diacritics, canonicalization)
- `indic_nlp.tokenize` — language-specific tokenization (respects scripts, splits correctly)

**What it does:**

The WAT evaluation protocol requires text to be tokenized using language-specific rules before BLEU computation:

```python
# Before (raw output):
text = "नमस्ते दोस्त"

# After tokenization (via Indic NLP):
text_tokenized = "नमस्ते दोस्त"  # splits on word boundaries, respects script rules

# Then BLEU is computed on tokenized version
```

**Used in:**
- `evaluate.py` — tokenizes predictions and references before BLEU/RIBES computation
- `data_utils.py::wat_tok()` — internal tokenization function

**Verify installation:**

```bash
python -c "from indicnlp import loader; loader.load_lang('hi'); print('✓ Indic NLP loaded successfully')"
```

If you see `✓ Indic NLP loaded successfully`, you're good to go.

**Troubleshooting:**

If installation fails:
```bash
# Alternative: Install via pip
pip install indic-nlp-library

# Then download resources separately:
cd ~/indic_nlp_library
git clone https://github.com/indicnlp/indic_nlp_resources.git
```

Then tell `evaluate.py` where to find it:
```bash
python evaluate.py \
    --lang hindi \
    --ckpt ./runs/hindi/best_model.pt \
    --data-dir ./data \
    --nllb-dir ./nllb13b_local \
    --indic-nlp ./indic_nlp_library
```

---

### Step 3 — Download the Dataset

Go to: **https://ufal.mff.cuni.cz/wat2025english-indicmultimodaltranslation**

Download **Hindi Visual Genome v1.1**. Extract and place inside `data/`. Files are already named correctly — no renaming needed.

**For training** (text files + image folders, ~4.5 GB total):

```
data/
├── hindi-visual-genome-train.txt               (28,930 lines)
├── hindi-visual-genome-train.images/           (28,928 images)
├── hindi-visual-genome-dev.txt                 (998 lines)
├── hindi-visual-genome-dev.images/             (998 images)
├── hindi-visual-genome-test.txt                (1,595 lines)
├── hindi-visual-genome-test.images/            (1,595 images)
├── hindi-visual-genome-challenge-test-set.txt  (1,400 lines)
└── hindi-visual-genome-challenge-test-set.images/  (1,400 images)
```

> **For evaluation and figures only** (no training), you only need the four `.txt` files (~4 MB). The `.images/` folders are NOT used at inference time.

**Verify line counts:**

```bash
wc -l data/hindi-visual-genome-train.txt                   # 28930
wc -l data/hindi-visual-genome-dev.txt                     # 998
wc -l data/hindi-visual-genome-test.txt                    # 1595
wc -l data/hindi-visual-genome-challenge-test-set.txt      # 1400
```

**File format** — 7 tab-separated columns, no header, raw pixel coordinates:

```
image_id   X     Y    W    H    English               Hindi
2376980    221   50   87   170  A stop light          एक स्टॉप लाइट
```

`data_utils.py` automatically normalizes X, Y, W, H to [0, 1] using 640×480 reference dimensions.

---

### Step 4 — Install System Dependencies (Linux/Ubuntu)

```bash
sudo apt-get install -y build-essential perl
```

- `perl`: Required for Moses multi-bleu evaluation script
- `build-essential`: Required for compiling Python packages

---

### Step 4.5 — Download Moses Decoder (for BLEU Evaluation)

Moses multi-bleu.perl is the official WAT evaluation metric. Download it:

```bash
git clone https://github.com/moses-smt/mosesdecoder.git
```

The script will be located at: `mosesdecoder/scripts/generic/multi-bleu.perl`

**Verify download:**

```bash
ls mosesdecoder/scripts/generic/multi-bleu.perl
# Should exist
```

This is **required for both training** (dev BLEU evaluation during training) **and evaluation**.

---

### Step 5 — Train

All paths are **fully customizable** via command-line arguments. Use defaults if files are in repo root; override if they're elsewhere.

**Single GPU (standard):**

```bash
python train.py --lang hindi \
    --data-dir ./data \
    --nllb-dir ./nllb13b_local \
    --indic-nlp ./indic_nlp_library
```

**Multi-GPU (recommended for faster training):**

```bash
torchrun --nproc_per_node=2 train.py --lang hindi \
    --data-dir ./data \
    --nllb-dir ./nllb13b_local \
    --indic-nlp ./indic_nlp_library \
    --moses ./mosesdecoder/scripts/generic/multi-bleu.perl
```

**Custom paths (if files are in different locations):**

```bash
torchrun --nproc_per_node=2 train.py --lang hindi \
    --data-dir /path/to/hindi-visual-genome-11 \
    --nllb-dir /path/to/nllb13b_local \
    --indic-nlp /path/to/indic_nlp_library \
    --moses /path/to/multi-bleu.perl \
    --work-dir /path/to/output
```

**All training arguments:**

| Argument | Default | Example | Purpose |
|----------|---------|---------|---------|
| `--lang` | hindi | `hindi`, `bengali` | Target language |
| `--data-dir` | `./data` | `/data/vg` | Dataset directory with TSV files |
| `--nllb-dir` | `./nllb13b_local` | `/models/nllb` | NLLB-1.3B weights directory |
| `--indic-nlp` | `./indic_nlp_library` | `/lib/indic` | Indic NLP Library directory |
| `--moses` | `./mosesdecoder/scripts/generic/multi-bleu.perl` | `/path/to/multi-bleu.perl` | Moses evaluation script (required) |
| `--work-dir` | `runs/{lang}` | `/output/hindi` | Checkpoint output directory |
| `--resume-from` | 0 | `15` | Resume from epoch (0=fresh) |
| `--resume-ckpt` | None | `/path/to/ckpt.pt` | Load weights from checkpoint |
| `--best-bleu` | -1.0 | `42.5` | Best dev BLEU before resuming |

**Training process:**
1. Trains for up to 30 epochs with early stopping (patience = 8 on dev BLEU)
2. Saves best checkpoint to `{work-dir}/best_model.pt` whenever dev BLEU improves
3. Evaluates on test + challenge splits and saves results to `{work-dir}/final_results.json`

**Resume an interrupted run:**

```bash
# If training was interrupted at epoch 20 with best BLEU=42.5
torchrun --nproc_per_node=2 train.py --lang hindi \
    --data-dir ./data \
    --nllb-dir ./nllb13b_local \
    --resume-from 20 --best-bleu 42.5
```
- Resumes from epoch 21
- Only saves new checkpoint if BLEU > 42.5
- Automatically loads `runs/hindi/best_model.pt`

**Expected hardware:** ~8 hours on a single A100 80 GB GPU.

---

### Step 6 — Evaluate

**Standard evaluation (default paths):**

```bash
python evaluate.py \
    --lang hindi \
    --ckpt ./runs/hindi/best_model.pt \
    --data-dir ./data \
    --nllb-dir ./nllb13b_local \
    --indic-nlp ./indic_nlp_library \
    --moses ./mosesdecoder/scripts/generic/multi-bleu.perl \
    --splits test challenge
```

**Custom paths:**

```bash
python evaluate.py \
    --lang hindi \
    --ckpt /path/to/best_model.pt \
    --data-dir /path/to/hindi-visual-genome-11 \
    --nllb-dir /path/to/nllb13b_local \
    --indic-nlp /path/to/indic_nlp_library \
    --moses /path/to/multi-bleu.perl \
    --splits test challenge
```

**Evaluation arguments:**

| Argument | Default | Purpose |
|----------|---------|---------|
| `--lang` | hindi | Target language: `hindi` or `bengali` |
| `--ckpt` | `runs/hindi/best_model.pt` | Path to trained checkpoint |
| `--data-dir` | `./data` | Dataset directory |
| `--nllb-dir` | `./nllb13b_local` | NLLB-1.3B weights directory |
| `--indic-nlp` | `./indic_nlp_library` | Indic NLP Library directory |
| `--moses` | `./mosesdecoder/scripts/generic/multi-bleu.perl` | Moses evaluation script (required) |
| `--splits` | `test challenge` | Splits to evaluate: `dev test challenge` |
| `--out-dir` | (repo root) | Directory to save results JSON |

**Expected output:**

```
Split        BLEU     RIBES
---------  ------  --------
test        44.67    0.8304
challenge   54.80    0.8624
```

Results also saved to `eval_results.json`.

---

## Training Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| LR KAN-MoE / RegionGate | 3 × 10⁻⁴ | New modules, higher LR |
| LR Decoder | 1 × 10⁻⁵ | Careful fine-tuning |
| LR Encoder | 1 × 10⁻⁵ | Mostly frozen, low LR |
| Effective batch size | 32 | (batch_size=8 × grad_accum=4) |
| Label smoothing | 0.1 | Regularization |
| Load-balance α | 0.05 | Switch auxiliary loss weight |
| Warmup steps | 300 | Linear warmup before cosine decay |
| Early stopping patience | 8 epochs | On dev BLEU |
| Training time | ~8 hrs | Single A100 80 GB |

All parameters defined in `config.py`. To reproduce exactly, keep all values unchanged.

---

## Model Architecture

```
RegionGate
  (x,y,w,h) → Fourier(n=16) → 128-d → MLP(2-layer) → 1024-d bias
  added to every NLLB encoder token (residual + LayerNorm)

KAN-MoE Fusion
  4 RBF-KAN experts (each: 2048 → 2048 dimensions)
  soft routing (all experts active per token)
  residual skip connection, load-balance loss α=0.05
  experts learn to specialize by part-of-speech without labels

NLLB-1.3B
  fully fine-tuned, no layer freezing
  discriminative learning rates: encoder 1e-5, decoder 1e-5, KAN/gate 3e-4
```

**Parameter breakdown:**

| Component | Params |
|-----------|--------|
| RegionGate | 2.36M |
| KAN-MoE | 272.78M |
| NLLB-1.3B | 2,157.7M |
| **Total** | **~2.43B** |

---

## Troubleshooting

### "Could not resolve host: github.com" or "No internet access"
- Download NLLB weights offline or pre-download them
- Use manual file download from HuggingFace web interface

### "Moses multi-bleu.perl not found"
- Ensure `perl` is installed: `sudo apt-get install perl`
- Verify script exists in evaluate.py paths

### "ImportError: transformers version mismatch"
- Use conda environment: `conda env create -f environment.yml`
- Or reinstall: `pip install transformers==4.36.2`

### GPU out of memory (OOM)
- Reduce `batch_size` in `config.py` (from 8 to 4 or 2)
- Reduce `grad_accum` (gradient accumulation steps)
- Use single GPU instead of multi-GPU

### BLEU scores much lower than expected
- Verify dataset TSV files have correct format (7 columns, tab-separated)
- Check bounding box normalization: should be [0, 1] range
- Ensure NLLB weights loaded correctly: `ls nllb13b_local/pytorch_model.bin`
- Try warmup: may need more epochs for convergence

---

## Citation

If you use KAN-MoE in your research, please cite the following paper:

```bibtex
@article{singh2025kan,
  title={KAN-MoE: Interpretable Expert Routing for Indic Machine Translation},
  author={Singh, Harsh Kumar and Gain, Baban and Kumar, Deepak and Ekbal, Asif},
  journal={arXiv preprint arXiv:2501.xxxxx},
  year={2025}
}
```

**Plain text citation:**

Singh, Harsh Kumar, Baban Gain, Deepak Kumar, and Asif Ekbal. "KAN-MoE: Interpretable Expert Routing for Indic Machine Translation." Preprint submitted to Elsevier, 2025.

**Key contributions cited in the paper:**
- KAN-based Mixture-of-Experts architecture with learnable RBF basis functions
- RegionGate spatial conditioning using Fourier-encoded bounding-box coordinates
- Interpretable expert routing with measurable bandwidth specialisation
- Empirical analysis showing POS-correlated expert specialisation without supervision
- Competitive performance on HindiVisualGenome benchmark (44.67 Test BLEU, 54.8 Challenge BLEU)

