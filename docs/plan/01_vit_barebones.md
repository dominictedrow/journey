# 01 — Barebones ViT Framework (v0 skeleton)

Status: **v0 barebones** (written 2026-09-24, before the research results were merged in).
The detailed, research-backed version of each section lives in `02_vit_detailed_plan.md`.
This file is the "shape" of the network: every component from Dom's list in CLAUDE.md is
placed and wired into the data flow. **No hyperparameters are fixed here**: all numbers are
placeholders (`HP:`) to tune later.

## 0. Design in one picture

A study is a **bag of series**. Each series is a **stack of 2D slices**. The model is a
*factorized* (two-level) encoder-only transformer:

```
study
 ├─ series 1 (e.g. Sagittal, fluid-sensitive, fat-sat)  ─┐
 ├─ series 2 (e.g. Coronal, PD)                          ├─ each slice → [SliceViT] → 1 slice token
 └─ series k ...                                         ─┘
                                   ↓
   slice tokens + slice-position emb + series-type emb (plane × fluid × fat-sup)
                                   ↓
                 [StudyTransformer] over all slice tokens of the study (+ [CLS])
                                   ↓
                     [Head] → 12 logits → sigmoid → BCE
```

Why factorized (in-plane first, then across slices/series) rather than one flat 3D ViT:
flat attention over every patch of every slice of every series is quadratic in a token count
of tens of thousands, which doesn't fit 12 GB of VRAM. Factorizing is the standard way around
this (ViViT "factorised encoder", TimeSformer divided attention). It's also CLAUDE.md's listed
"factorized space/slice attention" option. The alternatives stay as ablations (section 9).

## 1. Tensor contract (shapes)

| Symbol | Meaning | Placeholder |
|---|---|---|
| B | studies per batch | HP |
| S | series slots per study (padded, with a mask) | HP (e.g. up to 5) |
| D | slices sampled per series (padded, with a mask) | HP |
| C | channels per slice input: 1 (grayscale) or 3 (2.5D: slice−1, slice, slice+1) | ablation |
| H=W | slice size after resample/crop | HP |
| P | patch size | HP |
| N = (H/P)·(W/P) | patches per slice | derived |
| E | embedding dim | HP |

- Model input: `x: float32[B, S, D, C, H, W]`, `slice_mask: bool[B, S, D]`, `series_meta: long[B, S, 3]`
  (plane id, fluid flag, fat-sup flag), `series_mask: bool[B, S]`.
- Output: `logits: float32[B, 12]` (column order = submission order in CLAUDE.md).

## 2. Components (Dom's list → module → where it sits)

| # | Dom's component | Module (file) | Barebones behavior |
|---|---|---|---|
| 1 | Patch embedding → learned embedding matrix | `PatchEmbed` (`src/knee/model/patch_embed.py`) | `Conv2d(C, E, kernel=P, stride=P)` = a learned linear map of each flattened P×P patch (the two are mathematically identical). Flatten → `[.., N, E]`. |
| 2 | Positional encoding (patch position) | `PosEmbed2D` (`pos_embed.py`) | Learned `[1, N, E]` table added to patch tokens. Sinusoidal (fixed) variant behind a flag. |
| 3 | Slice position | `SlicePosEmbed` (`pos_embed.py`) | Encodes *where in the stack* a slice is (normalized 0..1 position, or index). Added to slice tokens before the StudyTransformer. |
| 4 | View / sequence embedding | `SeriesTypeEmbed` (`pos_embed.py`) | Embedding tables for plane (3), fluid (2), fat-sup (2), summed. Added to every slice token of that series. This is the "sequence type as conditioning input" item (checklist #9). |
| 5 | Learned [CLS] token | in `SliceViT` and `StudyTransformer` | One learned `[1,1,E]` token prepended at each level. The study-level CLS feeds the head. |
| 6 | Multi-head self-attention | `MultiHeadSelfAttention` (`attention.py`) | `qkv = Linear(E, 3E)` → split into h heads → `softmax(QKᵀ/√d)V` → merge heads → `Linear(E,E)`. Written by hand first, then swapped to `F.scaled_dot_product_attention` once it matches numerically. Takes a key-padding mask. |
| 7 | Pre-norm transformer block + residuals | `Block` (`block.py`) | `x = x + DropPath(Attn(LN(x)))`; `x = x + DropPath(MLP(LN(x)))`. |
| 8 | MLP with GELU | `Mlp` (`mlp.py`) | `Linear(E, r·E) → GELU → Dropout → Linear(r·E, E) → Dropout`. |
| 9 | LayerNorm (not BatchNorm) | inside `Block` + final norm | `nn.LayerNorm(E)` pre-attention, pre-MLP, and once after the last block. |
| 10 | Dropout / stochastic depth | `DropPath` (`block.py`), dropouts in `Mlp`/attn | Drop rate increases linearly with depth. All rates are HP. |
| 11 | Encoder only | whole model | No decoder. (An MAE *pretraining* decoder may be added later as a separate, throwaway module; see 02.) |
| 12 | Head: 12 independent logits, sigmoid + BCE | `MultiLabelHead` (`heads.py`) | `LN → Linear(E, 12)`. Loss `BCEWithLogitsLoss(pos_weight=HP)`. No softmax. |
| 13 | Optimizer AdamW, warmup + cosine, grad clipping | `src/knee/train/optim.py` | AdamW with no weight decay on norms, biases, pos-embeds and CLS; linear warmup → cosine decay; `clip_grad_norm_`. |
| 14 | Pos-weight / class-balanced sampling | `src/knee/train/sampling.py` | Either `pos_weight` per label in the BCE or a weighted sampler. Ablation. |
| 15 | Augmentation: spatial, intensity, slice-dropout ("temporal") | `src/knee/data/augment.py` | Spatial (small affine, crop jitter; **flips gated per finding, see 02**), intensity (gamma, contrast, noise), slice dropout (randomly mask slice tokens). |
| 16 | Efficient attention variants | `attention.py` (flag) | Factorization is already the main efficiency tool. Flash/SDPA kernel after the hand-written version is verified. Windowed / linear / token-merging are later options. |
| 17 | Quantization | `src/knee/infer/quantize.py` | Post-training only, after accuracy is set: fp16 autocast first, then int8 (dynamic/PTQ) if it helps runtime on the Kaggle GPU. |
| 18 | Multi-sequence fusion: channels vs token streams | `StudyTransformer` input builder | **Default = token streams** (each series contributes its own slice tokens, tagged by `SeriesTypeEmbed`). Channel stacking stays an ablation. |
| 19 | Series metadata (plane, fluid, fat-sup; demographics only if present) | `SeriesTypeEmbed` (+ optional `MetaEmbed`) | See #4. Demographics only if the tags exist (TODO in CLAUDE.md). |
| 20 | Single-channel grayscale per slice | `PatchEmbed(C=1)` | Default C=1. C=3 (2.5D neighbor slices) is an ablation. |

## 3. Module skeletons (signatures only — implementation happens in the build sessions)

```python
# src/knee/model/patch_embed.py
class PatchEmbed(nn.Module):
    def __init__(self, in_ch: int, embed_dim: int, patch: int): ...
    def forward(self, x):            # [N_img, C, H, W] -> [N_img, N_patches, E]
        ...

# src/knee/model/pos_embed.py
class PosEmbed2D(nn.Module):        # learned (default) or sincos
    def forward(self, tokens): ...  # adds [1, N, E]
class SlicePosEmbed(nn.Module):
    def forward(self, tokens, slice_pos): ...   # slice_pos: float[B,S,D] in [0,1]
class SeriesTypeEmbed(nn.Module):
    def forward(self, tokens, series_meta): ... # plane/fluid/fatsup lookups, summed

# src/knee/model/attention.py
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads, attn_drop=0., proj_drop=0., use_sdpa=False): ...
    def forward(self, x, key_padding_mask=None): ...

# src/knee/model/mlp.py
class Mlp(nn.Module):
    def __init__(self, dim, ratio, drop): ...

# src/knee/model/block.py
class DropPath(nn.Module): ...
class Block(nn.Module):             # pre-norm
    def __init__(self, dim, heads, mlp_ratio, drop, attn_drop, drop_path): ...
    def forward(self, x, key_padding_mask=None): ...

# src/knee/model/slice_vit.py
class SliceViT(nn.Module):
    """2D ViT over one slice (or one 2.5D window). Returns one E-dim vector per slice (CLS)."""
    def forward(self, x):            # [N_img, C, H, W] -> [N_img, E]
        ...

# src/knee/model/study_transformer.py
class StudyTransformer(nn.Module):
    """Transformer over the study's slice tokens across all series, with its own CLS."""
    def forward(self, slice_tokens, slice_pos, series_meta, token_mask): ...  # -> [B, E]

# src/knee/model/heads.py
class MultiLabelHead(nn.Module):
    def forward(self, z): ...        # [B, E] -> [B, 12] logits

# src/knee/model/knee_vit.py
class KneeViT(nn.Module):
    def forward(self, x, slice_mask, series_meta, series_mask, slice_pos): ...
        # 1. flatten valid slices -> SliceViT -> [B,S,D,E]
        # 2. + SlicePosEmbed + SeriesTypeEmbed
        # 3. flatten S*D -> StudyTransformer (mask padding) -> CLS -> [B,E]
        # 4. MultiLabelHead -> [B,12]
```

## 4. Data flow around the model (one line each; detail in 02 and the preprocessing plan)

1. DICOM → decode by transfer syntax → float32 → modality LUT (slope/intercept) → sort slices by position.
2. Per-series intensity normalization (z-score or percentile clip; ablation).
3. Resample/crop to a common physical field of view → `H×W`; sample `D` slices per series.
4. Choose up to `S` series per study by (plane, fluid, fat-sup) rules; pad and mask missing ones.
5. Cache to compact uint8/uint16 arrays on a fast internal drive. **The same code runs at inference.**
6. Dataset → augment → collate `[B,S,D,C,H,W]` + masks + metadata.
7. Model → logits → BCE vs (soft) labels → AdamW step.
8. Validation: macro AUC on the held-out split(s). Split **by study**; also group by site if it can be inferred.

## 5. Sanity tests that gate each component (write before or with the code)

- `PatchEmbed`: the Conv2d version equals the manual unfold+Linear version (allclose).
- `MultiHeadSelfAttention`: the hand-written version equals `F.scaled_dot_product_attention` and `nn.MultiheadAttention` for the same weights; masking really zeroes padded keys.
- `Block`: output shape preserved; gradients reach all params.
- `KneeViT`: loss at init ≈ 0.693 per label (zero-init head); overfits one batch to ~0 loss; permuting padded slices doesn't change the output.
- End to end: a random-label run gives AUC ≈ 0.5; a shuffled-image run falls to ≈ 0.5.

## 6. Training stages (barebones order)

1. **Pipeline smoke test** on a few hundred studies: the model trains and the loss falls.
2. **Supervised from scratch** on the label set chosen in the labeling plan: first real baseline.
3. **Self-supervised pretraining** of `SliceViT` on all slices (added from research; see 02): the likely key to making a scratch ViT competitive.
4. Fine-tune the full `KneeViT`; tune; ablate.
5. Shrink for efficiency (fp16 → quantization → fewer slices/series), measuring AUC vs runtime.

## 7. Inference skeleton

`notebooks/infer.ipynb`: offline wheels → read `test_series.csv` → same preprocessing → batched
fp16 forward → sigmoid → `submission.csv` in the exact column order. Every study must get a row;
fall back to 0.5 or the train prior on any failure, and log it.

## 8. What is deliberately NOT decided here

Model size (E, depth, heads, P, H, D, S), 1 vs 3 channels, learned vs sincos positions,
CLS vs attention pooling at study level, loss weighting, augmentation magnitudes, SSL method.
These are listed in `02_vit_detailed_plan.md` as HP/ablations with the research that informs them.

## 9. Ablation hooks built into the skeleton from day one

- `fusion = {"tokens", "channels"}`
- `in_ch = {1, 3}` (2.5D)
- `pos2d = {"learned", "sincos"}`
- `study_pool = {"cls", "attn_per_label"}`
- `attn_impl = {"manual", "sdpa"}`
- `flip_aug = per-finding gate`
