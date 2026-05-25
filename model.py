
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput

from config import CFG


class KANExpert(nn.Module):
    """
    RBF-KAN expert: learns basis centers and widths per dimension.
    Fixed: dropout applied before projection (not after), matching original model.py.
    """
    def __init__(self, d_in: int, d_out: int, n_basis: int, dropout: float = 0.1):
        super().__init__()
        self.d_in    = d_in
        self.n_basis = n_basis
        self.centers   = nn.Parameter(torch.randn(n_basis, d_in) * 0.02)
        self.log_sigma = nn.Parameter(torch.zeros(n_basis))
        self.out_proj  = nn.Linear(n_basis * d_in, d_out, bias=True)
        self.dropout   = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        diff  = x.unsqueeze(2) - self.centers.unsqueeze(0).unsqueeze(0)
        sigma = self.log_sigma.exp().clamp(min=1e-4)
        phi   = torch.exp(-0.5 * (diff / sigma.view(1, 1, -1, 1)) ** 2)
        out   = phi.reshape(B, T, self.n_basis * D)
        return self.out_proj(self.dropout(out))


class KANMoEFusion(nn.Module):
    """
    KAN Mixture-of-Experts with residual connection and correct Switch-style aux loss.

    Fixes vs old model_nllb13b.py:
      1. Residual skip: output = LayerNorm(h_in + out_proj(mixed))
      2. aux_loss = n_experts * sum(f_bar^2)  [Switch Transformer formulation]
         not (f_e * P_e) where both were the same soft-mean — that double-counted.
      3. out_proj added to compress kan_hidden → d_model with skip.
    """
    def __init__(self, d_model: int, kan_hidden: int, n_experts: int,
                 n_basis: int, dropout: float = 0.1):
        super().__init__()
        self.n_experts = n_experts
        self.experts   = nn.ModuleList([
            KANExpert(kan_hidden, kan_hidden, n_basis, dropout)
            for _ in range(n_experts)
        ])
        self.in_proj  = nn.Linear(d_model, kan_hidden)
        self.gate     = nn.Linear(kan_hidden, n_experts)
        self.out_proj = nn.Linear(kan_hidden, d_model)
        self.norm     = nn.LayerNorm(d_model)
        self.dropout  = nn.Dropout(dropout)
        self.last_gates: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> tuple:
        h = self.in_proj(x)                                    # (B, T, kan_hidden)
        gates = F.softmax(self.gate(h), dim=-1)                # (B, T, n_experts)
        self.last_gates = gates.detach().cpu()

        expert_outs = torch.stack([e(h) for e in self.experts], dim=2)
        mixed = (gates.unsqueeze(-1) * expert_outs).sum(dim=2) # (B, T, kan_hidden)

        # Correct Switch aux loss: L = n_experts * sum(f_bar^2)
        f_bar    = gates.mean(dim=[0, 1])                       # (n_experts,)
        aux_loss = self.n_experts * (f_bar ** 2).sum()

        out = self.out_proj(self.dropout(mixed))                # (B, T, d_model)
        return self.norm(x + out), aux_loss                     # residual + norm


class RegionGate(nn.Module):
    """
    Fourier spatial conditioning: bbox → gated additive bias per token.

    Fixed vs old version:
      - Returns norm(h + gate_bias) instead of norm(h * scale + shift).
        The multiplicative form could suppress useful encoder features.
        Additive bias is safer and still expressive.
      - Fourier features are computed once and shared between gate and shift.
    """
    def __init__(self, d_model: int, n_freqs: int = 16):
        super().__init__()
        in_dim = 4 * 2 * n_freqs   # 128
        self.gate_proj = nn.Sequential(
            nn.Linear(in_dim, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.n_freqs = n_freqs

    def _fourier(self, bbox: torch.Tensor) -> torch.Tensor:
        feats = []
        for i in range(4):
            u = bbox[:, i]
            for k in range(self.n_freqs):
                s = 2.0 ** k
                feats.append(torch.sin(s * torch.pi * u))
                feats.append(torch.cos(s * torch.pi * u))
        return torch.stack(feats, dim=-1)                       # (B, 128)

    def forward(self, h: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
        phi  = self._fourier(bbox)                              # (B, 128)
        bias = self.gate_proj(phi).unsqueeze(1)                 # (B, 1, d_model)
        return self.norm(h + bias)                              # additive residual


class KANMoENLLB(nn.Module):
    """
    NLLB-1.3B + KAN-MoE + RegionGate (v2, architecture fixes).

    Pipeline:
      src_text → NLLB-1.3B encoder → h_enc
      h_enc    → RegionGate(bbox)  → h_spatial   (additive Fourier bias first)
      h_spatial → KAN-MoE          → h_fused     (residual, correct aux loss)
      h_fused  → NLLB-1.3B decoder → translation

    Key changes vs model_nllb13b.py:
      1. RegionGate BEFORE KAN-MoE (spatial context helps expert routing).
      2. KANMoEFusion has in_proj + residual skip + out_proj (matches original design).
      3. Correct Switch aux loss formula.
      4. Additive RegionGate instead of multiplicative (safer, preserves features).
      5. kan_hidden = 2048 (2x d_model) for richer expert representations.
    """
    def __init__(self, cfg: CFG):
        super().__init__()
        self.cfg = cfg

        self.nllb13b = AutoModelForSeq2SeqLM.from_pretrained(cfg.nllb13b_local)

        kan_hidden = cfg.kan_hidden   # set to 2048 in config for v2
        self.region_gate = RegionGate(d_model=cfg.d_model)
        self.kan_moe = KANMoEFusion(
            d_model=cfg.d_model, kan_hidden=kan_hidden,
            n_experts=cfg.n_experts, n_basis=cfg.kan_basis,
            dropout=cfg.kan_dropout,
        )

    def _build_encoder_states(self, input_ids, attention_mask, bbox):
        enc_out = self.nllb13b.model.encoder(
            input_ids=input_ids, attention_mask=attention_mask,
        )
        h_enc     = enc_out.last_hidden_state
        h_spatial = self.region_gate(h_enc, bbox)     # spatial first
        h_fused, aux_loss = self.kan_moe(h_spatial)   # then KAN-MoE with residual
        return h_fused, attention_mask, aux_loss

    def forward(self, input_ids, attention_mask, labels, bbox,
                label_smoothing=0.1, **kwargs):
        h_enc, mask_enc, aux_loss = self._build_encoder_states(
            input_ids, attention_mask, bbox,
        )
        out = self.nllb13b(
            input_ids=input_ids,
            attention_mask=mask_enc,
            labels=labels,
            encoder_outputs=BaseModelOutput(last_hidden_state=h_enc),
        )

        if label_smoothing > 0:
            logits    = out.logits
            V         = logits.size(-1)
            log_probs = F.log_softmax(logits, dim=-1)
            mask      = labels != -100
            nll    = -log_probs.gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            smooth = -log_probs.sum(dim=-1) / V
            loss   = ((1 - label_smoothing) * nll + label_smoothing * smooth)
            loss   = loss[mask].mean()
        else:
            loss = out.loss

        return loss + self.cfg.moe_aux_wt * aux_loss, aux_loss.detach()

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, bbox, tokenizer, **kwargs):
        h_enc, mask_enc, _ = self._build_encoder_states(input_ids, attention_mask, bbox)
        forced_bos = tokenizer.convert_tokens_to_ids(self.cfg.tgt_lang_nllb)
        out_ids = self.nllb13b.generate(
            input_ids=input_ids,
            attention_mask=mask_enc,
            encoder_outputs=BaseModelOutput(last_hidden_state=h_enc),
            forced_bos_token_id=forced_bos,
            max_new_tokens=self.cfg.max_tgt_len,
            num_beams=self.cfg.beam_size,
            length_penalty=self.cfg.length_penalty,
            no_repeat_ngram_size=self.cfg.no_repeat_ngram,
            repetition_penalty=self.cfg.repetition_penalty,
        )
        return tokenizer.batch_decode(out_ids, skip_special_tokens=True)

    def get_param_groups(self):
        seen = set()
        def dedup(params):
            out = []
            for p in params:
                if id(p) not in seen:
                    seen.add(id(p))
                    out.append(p)
            return out

        new_params = dedup(
            list(self.kan_moe.parameters()) +
            list(self.region_gate.parameters())
        )
        decoder_params = dedup(
            list(self.nllb13b.model.decoder.parameters()) +
            list(self.nllb13b.lm_head.parameters())
        )
        encoder_params = dedup(
            list(self.nllb13b.model.encoder.parameters())
        )
        return [
            {"params": new_params,     "lr": self.cfg.lr_kan,     "name": "kan_moe_region_gate"},
            {"params": decoder_params, "lr": self.cfg.lr_decoder, "name": "decoder"},
            {"params": encoder_params, "lr": self.cfg.lr_encoder, "name": "encoder"},
        ]

    def unfreeze_encoder_layers(self, from_top: int, to_top: int, lr: float):
        """
        Unfreeze encoder layers[-to_top : -from_top] (exclusive of already-unfrozen top-from_top).
        Call with from_top=0, to_top=4 first; then from_top=4, to_top=8 next.
        This guarantees zero parameter overlap between successive calls.
        """
        layers = self.nllb13b.model.encoder.layers
        if from_top == 0:
            target_layers = layers[-to_top:]
        else:
            target_layers = layers[-to_top:-from_top]
        params = []
        for layer in target_layers:
            for p in layer.parameters():
                p.requires_grad_(True)
                params.append(p)
        return {"params": params, "lr": lr, "name": f"nllb13b_enc_top{to_top}"}

    def param_counts(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
