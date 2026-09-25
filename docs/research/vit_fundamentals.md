# ViT-from-scratch fundamentals — component-by-component, with emphasis on small/medium data

Scope: what it takes to implement a Vision Transformer by hand in PyTorch (no `timm.create_model`)
and train it well on **~7,500 studies (58 verified-labeled)**, on a 12 GB VRAM / 16 GB RAM box.
Every non-trivial numeric claim is cited. Anything not confirmed from a primary source is marked
`UNVERIFIED`.

---

## 1. Patch embedding

**Linear projection == strided Conv2d.** In the original ViT, an image is split into fixed-size
patches, each patch is flattened and passed through a shared linear layer. Splitting into
non-overlapping `P×P` patches and applying one `Linear(P*P*C → D)` is mathematically identical to a
`Conv2d(in_channels=C, out_channels=D, kernel_size=P, stride=P)` — the conv kernel *is* the linear
weight matrix, reshaped, applied at every non-overlapping window. This is exactly how the official
HuggingFace `timm` implementation does it: `PatchEmbed` is `nn.Conv2d(in_chans, embed_dim,
kernel_size=patch_size, stride=patch_size)` followed by a flatten
(`timm/layers/patch_embed.py`, https://github.com/huggingface/pytorch-image-models). lucidrains'
`vit-pytorch` instead does the literal einops-rearrange-then-Linear version (`Rearrange('b c (h p1)
(w p2) -> b (h w) (p1 p2 c)')` → `LayerNorm` → `Linear` → `LayerNorm`,
https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py) — same math, and it adds
two LayerNorms around the projection (not in the original paper, but common practice in that repo)
for stability. Either formulation is fine; Conv2d is more common and slightly more efficient in
PyTorch.

**Patch size trade-offs.** Original ViT compares 14×14, 16×16, 32×32 patches; smaller patches ⇒
more tokens ⇒ quadratically more compute in self-attention but finer spatial resolution and
(empirically) better accuracy — ViT-L/16 beats ViT-L/32 substantially in the paper
(arXiv:2010.11929, https://arxiv.org/abs/2010.11929). For medical images where small findings
(fractures, small tears) matter, favor smaller patches, but this directly multiplies token count and
therefore attention FLOPs/memory (O(N²)) — a first-order VRAM-budget knob on a 12 GB card.

**Convolutional stems help optimization and data-efficiency.** "Early Convolutions Help
Transformers See Better" (Xiao et al. 2021, arXiv:2106.14881,
https://ar5iv.labs.arxiv.org/html/2106.14881) replaces the single stride-16 "patchify" conv with a
stack of **4–6 stacked 3×3 stride-2 convolutions** (e.g. channel progression `[24,48,96,192]` for
their 1GF model), keeping FLOPs roughly matched by removing one transformer block. Findings directly
relevant to us:
- **Optimizer robustness**: the standard patchify stem loses **7–10% top-1 accuracy** switching from
  AdamW to SGD; the conv-stem version's AdamW/SGD gap shrinks to **<0.2%** — i.e. the conv stem makes
  the model much less sensitive to optimizer/LR choice, which matters a lot when you can't afford a
  huge LR sweep.
- **Faster convergence**: at 50 epochs, the conv-stem model is only 6% (relative) away from its
  400-epoch asymptote vs. 10% for the patchify version — meaningful when compute-constrained.
- **Accuracy**: +1–2% top-1 across model sizes, and this gap persists even with large-scale
  (ImageNet-21k) pretraining.
This is a strong, cheap, low-risk addition for a from-scratch ViT trained on a small/medium dataset:
it injects some convolutional locality bias exactly where ViTs lack it, at the input.

---

## 2. Positional encodings

| Scheme | Source | Notes |
|---|---|---|
| Learned 1D | Original ViT (arXiv:2010.11929) | Flattened patch sequence gets one learned vector per position, added to patch embeddings. Ablation in the ViT paper (Table 8, ViT-B/16, 5-shot ImageNet linear eval): **no pos-emb 0.6138, 1-D 0.6421, 2-D 0.6400, relative 0.6403** — the paper's conclusion is *"there is little to no difference between different ways of encoding positional information"* once you have positional information of some kind. The jump from none → any encoding is the big effect. |
| 2D sin-cos (fixed, not learned) | MAE (He et al. 2021, arXiv:2111.06377) | MAE states: *"Our MAE adds positional embeddings (the sine-cosine version) to both the encoder and decoder inputs"* — this is the standard Transformer sinusoidal formula extended to a 2D grid (separate sin/cos bands for row and column, concatenated). MoCo v3 uses the same idea (UNVERIFIED exact citation location, but widely used alongside MAE for self-supervised ViT pretraining). **"Better plain ViT baselines"** (Beyer, Zhai, Kolesnikov 2022, arXiv:2205.01580) directly ablates fixed sincos2d vs. learned and finds sincos2d *better*, with the gap larger at low epoch counts: 90 epochs 76.5%→75.0%, 150 epochs 78.5%→78.0%, 300 epochs 80.0%→79.6% (sincos2d → learned). **Practical takeaway: for training from scratch on a small compute/epoch budget, fixed 2D sincos is a free accuracy gain over learned 1D and needs zero extra parameters.** |
| Relative position bias | ViT ablation (Table 8 above), Swin-style | Comparable to 1D/2D absolute in the ViT ablation; more commonly used in windowed architectures (Swin) than plain ViT. |
| RoPE for vision (RoPE-ViT, 2024) | Heo et al., ECCV 2024, arXiv:2403.13298, https://arxiv.org/abs/2403.13298; code https://github.com/naver-ai/rope-vit | Extends rotary position embedding (originally for language-model attention) to 2D image grids in two flavors: **axial RoPE** (independent rotation per axis) and **mixed RoPE** (learnable frequency mixing across axes). Reported DeiT-III-trained ImageNet-1k top-1 (400 epochs, 224×224, from repo README): DeiT-S baseline 80.4% → RoPE-axial 80.9% → RoPE-mixed 80.9%; DeiT-B baseline 83.4% → RoPE-axial 83.6% → RoPE-mixed 83.8%. The headline benefit is **extrapolation**: RoPE-ViT models keep much more accuracy when evaluated at resolutions larger than training resolution than absolute-pos-embedding models do (paper abstract). The repo also ships `rope_*_ape` hybrid variants that combine RoPE with absolute pos-embeddings, implying RoPE and APE are not mutually exclusive. |
| Interpolation for different input sizes | ViT paper (arXiv:2010.11929) + `timm` | The ViT paper performs 2D bicubic interpolation of the pretrained 1D learned position embeddings (reshaped to the original grid) when fine-tuning at a different resolution than pretraining. `timm` exposes this directly as `resample_abs_pos_embed` (seen imported in `timm/models/vision_transformer.py`). This matters for us if we ever want to fine-tune at a higher resolution after initial training, or handle series with a variable slice/patch grid. |

**Which works best from scratch, for us:** Given a small labeled set and modest compute, prefer
**fixed 2D sin-cos** (Beyer et al. show it beats learned even with pure supervised training, zero
extra params, zero extra risk) as the default; treat **RoPE-mixed** as a promising ablation if we
later care about extrapolating to different slice counts/resolutions across series, since our slice
count varies per series (median 30, long tail to hundreds per `CLAUDE.md`).

---

## 3. [CLS] token vs. global average pooling / attention pooling

**"Better plain ViT baselines for ImageNet-1k"** (Beyer, Zhai, Kolesnikov, 2022, arXiv:2205.01580,
https://ar5iv.labs.arxiv.org/html/2205.01580) directly ablates GAP vs. CLS token on top of an
otherwise-identical recipe (Table 1): GAP beats CLS at every epoch budget — **90 epochs: 76.5% (GAP)
vs 75.0% (CLS); 150 epochs: 78.5% vs 76.9%; 300 epochs: 80.0% vs 78.2%.** Their explanation
(paraphrased from the paper) is that the extra CLS-token learning-rate/weight-decay sensitivity (it's
a "bag of hacks" interacting with the rest of the network) is not worth it — plain GAP is simpler and
better in their setup. Their full recommended recipe drops CLS in favor of GAP entirely and also
drops dropout/stochastic depth (§6 below).

**Attention pooling / multi-query pooling heads** generalize GAP to a learned, query-based
weighted pool, and are directly relevant to a **12-label multi-output problem**:
- **Query2Label** (arXiv:2107.10834, https://ar5iv.labs.arxiv.org/html/2107.10834): one learnable
  query embedding *per label class*, run through a Transformer-decoder cross-attention layer against
  the patch/token features, so each label gets its own spatially-adaptive pooled feature instead of
  sharing one global vector. Reported gains: 91.3% mAP on MS-COCO, 96.6% mAP on Pascal VOC 2012 (at
  larger backbones/resolutions than we'd use, so treat as directional, not as an expected absolute
  number for us).
- **ML-Decoder** (arXiv:2111.12933, https://ar5iv.labs.arxiv.org/html/2111.12933): the same idea but
  with a fixed, smaller number of *group* queries (K ≪ num_classes) that get linearly expanded to the
  N classes via a "group fully-connected" layer, plus **removal of self-attention among queries**
  (justified because a linear projection can already map any fixed query to any effective query).
  This is *more efficient* than Query2Label at large label counts (their headline case is Open
  Images' 9,600 classes) but for only 12 labels, that efficiency argument barely matters — plain
  Query2Label-style (12 explicit queries, one per finding) is simpler to implement and reason about.
- For our 12 targets, **"one query per label" cross-attention pooling is a natural fit**: it lets
  e.g. the "Baker's cyst" query attend to the posterior joint region and the "ACL" query attend to
  the intercondylar notch region, instead of forcing one CLS/GAP vector to represent all 12 findings
  simultaneously. This is an ablation-worthy upgrade over plain GAP, not a must-have on day one.

**Recommendation:** start with **GAP** (simplest, proven better than CLS from scratch, per Beyer et
al.), and treat a **12-query cross-attention pooling head** (Query2Label-style) as the first
"beyond-baseline" architectural upgrade once GAP+linear-head is working and validated.

---

## 4. Transformer block internals

**Pre-norm vs. post-norm.** The original Transformer used post-norm (`LayerNorm(x + Sublayer(x))`);
essentially all modern ViT implementations use **pre-norm** (`x + Sublayer(LayerNorm(x))`) because it
gives much more stable gradients for deep stacks and doesn't need learning-rate warmup to avoid
divergence to the same degree. Both `timm` and `lucidrains/vit-pytorch` implement pre-norm blocks:
`x = x + drop_path(attn(norm1(x))); x = x + drop_path(mlp(norm2(x)))` — see
`timm/models/vision_transformer.py` `Block.forward` and `lucidrains`' `Transformer.forward`
(`x = attn(x) + x; x = ff(x) + x` where `attn`/`ff` apply their own internal pre-norm).

**LayerNorm eps.** PyTorch's `nn.LayerNorm` default eps is `1e-5`. `timm`'s ViT model factory
functions explicitly set `norm_layer=partial(LayerNorm, eps=1e-6)` for essentially all its standard
ViT configs (grep of `timm/models/vision_transformer.py` shows `eps=1e-6` used throughout the
model-builder functions, though the differential-attention sub-norm uses `eps=1e-5`) — `1e-6` is the
common ViT-community default, mirroring the original JAX implementation
(https://github.com/google-research/vision_transformer). Use `eps=1e-6` for the ViT body.

**MHSA with `F.scaled_dot_product_attention` (SDPA / FlashAttention).** `timm`'s `Attention` class
checks `use_fused_attn()` and, when available, calls
`F.scaled_dot_product_attention(q, k, v, dropout_p=..., is_causal=...)` instead of the manual
`softmax(qkᵀ·scale)@v` — this dispatches to FlashAttention-family fused kernels on supported
hardware, reducing memory from O(N²) materialized attention matrices to a fused/tiled computation,
and is meaningfully faster and more memory-efficient (`timm/layers/attention.py`,
https://github.com/huggingface/pytorch-image-models). On a 12 GB RTX 5070, using SDPA rather than a
hand-rolled `attn = softmax(q@k.T)` is a straightforward way to fit more tokens/batch size — worth
using SDPA directly (`torch.nn.functional.scaled_dot_product_attention`) even in a "from scratch"
implementation, since it's a PyTorch built-in, not a third-party model API — writing your own
softmax-attention manually is a reasonable *first* version for understanding, then swap to SDPA for
training-time efficiency once correctness is verified (see debugging §9).

**QK-norm (ViT-22B).** "Scaling Vision Transformers to 22 Billion Parameters" (Dehghani et al. 2023,
arXiv:2302.05442, https://ar5iv.labs.arxiv.org/html/2302.05442) found that at ~8B parameters, training
loss diverged after a few thousand steps because attention logits grew unboundedly (near-one-hot,
near-zero-entropy attention), with logit magnitude exceeding 50,000 within 2,000 steps without
mitigation. Their fix: apply **LayerNorm to the query and key projections independently**, before the
dot product: `softmax(LN(XW^Q)·LN(XW^K)ᵀ / √d)`. This let them use a **constant LR of 1e-3 across all
model scales**, instead of needing to shrink LR as the model grows. `timm` implements this as the
`qk_norm` flag on `Attention`/`Block` (`timm/layers/attention.py`,
`timm/models/vision_transformer.py`). **For our model sizes (Tiny/Small, §8), QK-norm is unlikely to
be strictly necessary — instability at that scale is a large-model phenomenon — but it is cheap
(two extra LayerNorms per block) and a reasonable stabilizer to include from the start, especially if
training in fp16/bf16 where attention-logit blowup is more likely to cause NaNs.**

**MLP ratio, GELU.** Standard ViT MLP block: `Linear(D → 4D) → GELU → Linear(4D → D)`, i.e. **MLP
ratio = 4** (confirmed by the ViT-Base/Large/Huge hidden sizes vs MLP sizes: 768/3072, 1024/4096,
1280/5120 — exactly ×4 — arXiv:2010.11929). `timm`'s `Block.__init__` default is
`mlp_ratio: float = 4.`, confirming this remains the community default.

**LayerScale (CaiT).** "Going Deeper with Image Transformers" (Touvron et al. 2021, arXiv:2103.17239,
https://ar5iv.labs.arxiv.org/html/2103.17239) introduces a **per-channel learnable diagonal scale**
on each residual branch's output before it's added back:
`x' = x + diag(λ_1,...,λ_d) · SA(LN(x))`, and similarly for the FFN branch — a *vector*, not a
scalar, unlike ReZero/Fixup. Initialization is **depth-dependent**: `ε=0.1` for ≤18 layers, `ε=1e-5`
for 24 layers, `ε=1e-6` for >24 layers. The intent is to start every residual branch's contribution
near-zero so the network initially behaves close to identity, and let training gradually "turn on"
each block — this stabilizes very deep ViT training. `timm`'s `Block` implements this as
`LayerScale(dim, init_values=init_values)` applied via `ls1`/`ls2` wrapping the attn/mlp output before
`drop_path`, controlled by the `init_values` constructor arg (`nn.Identity()` if `None`).
**For our shallow/medium models (§8, likely ≤12 layers), LayerScale is lower priority than for
CaiT's 24–36+ layer regime, but it's cheap and harmless to include — set `init_values≈0.1` if used at
our depth per the CaiT depth-conditioned schedule.**

**Stochastic depth / DropPath.** Implements per-sample residual-branch dropping: with probability
`drop_prob`, skip the entire attention or MLP sub-block for that sample (scaled by `1/(1-drop_prob)`
to keep expectation, controlled by `scale_by_keep`) — `timm/layers/drop.py`, class `DropPath`. The
standard schedule is **linear**: rate 0 at the first block, increasing linearly to
`drop_path_rate` at the last block — `timm`'s `calculate_drop_path_rates(drop_path_rate, depths,
stagewise=False)` implements exactly this "linear increase from 0 to `drop_path_rate` across all
blocks" pattern. DeiT (arXiv:2012.12877) uses **stochastic depth 0.1** in its base 300-epoch recipe
(Table 9, see §6); DeiT III (arXiv:2204.07118) scales it up with model size: **ViT-B: 0.1, ViT-L:
0.4, ViT-H: 0.5**.

**Dropout.** Standard (non-stochastic-depth) dropout on attention weights and MLP activations.
Notably, **"Better plain ViT baselines"** found dropout and stochastic depth both *unnecessary* at
their epoch budgets when data augmentation (RandAugment + Mixup) alone was used
(arXiv:2205.01580 — *"standard data augmentation is sufficient"*, no dropout or stochastic depth in
their final recipe). The original ViT paper used **dropout = 0.0** during large-scale (JFT-300M)
pretraining (Table 3, arXiv:2010.11929) but did use light dropout when fine-tuning on smaller
downstream sets — dropout's value is heavily regime-dependent (see §6, §7).

---

## 5. Initialization

**`trunc_normal_(std=0.02)`.** `timm`'s default ("timm-style") init for every `nn.Linear.weight`
is `trunc_normal_(weight, std=.02)`, biases zero-initialized
(`init_weights_vit_timm` in `timm/models/vision_transformer.py`). This matches BERT/GPT-style
initialization and is the most commonly copied init scheme for from-scratch ViTs (used by
`lucidrains/vit-pytorch` implicitly via PyTorch defaults being close, and explicitly in most
from-scratch ViT tutorials).

**JAX-style init (`init_weights_vit_jax`, matches the official Google implementation
https://github.com/google-research/vision_transformer).** Different rule for the classification
head vs. the body: **the head's `Linear.weight` is zero-initialized and its bias set to a
configurable `head_bias` (default 0.0)**; all other `Linear` layers get **Xavier/Glorot uniform**
init, with MLP-layer biases getting `normal_(std=1e-6)` and everything else zero bias; `Conv2d`
(the patch-embed projection) gets **LeCun-normal** init. This is directly relevant to us:

**Why zero-init (or near-zero) the head matters.** If the final classification `Linear`'s weight is
zero and bias is zero, the model's logits at initialization are exactly 0 for every class →
sigmoid(0) = 0.5 for every one of our 12 labels → **BCEWithLogits loss at init ≈ `-log(0.5) =
0.6931` per label** (see §9 — this is the sanity-check number to watch for). Zero-initializing the
head also means the *randomly initialized backbone's* garbage features don't get amplified into
wild initial logits, giving training a clean, well-behaved starting point and avoiding the "hockey
stick" loss curve Karpathy warns about (https://karpathy.github.io/2019/04/25/recipe/ — *"Initialize
the final layer weights correctly... this eliminates misleading hockey stick loss curves"*).

**MoCo v3-style init** (`init_weights_vit_moco` in `timm`) is a third variant: `qkv` weights get a
special uniform init scaled by `sqrt(6/(dim_in/3 + dim_out))` (essentially a Glorot-uniform variant
that accounts for the fact that a fused `qkv` weight matrix is really three separate projections),
everything else Xavier-uniform, zero bias. Included in `timm` for reproducing MoCo v3 pretraining;
not necessary for us to replicate exactly, but shows that treating fused QKV projections' fan-in/out
correctly at init is a known detail worth not overlooking if implementing a fused-QKV linear layer.

**Positional embedding / class token init.** `timm`'s `VisionTransformer.init_weights` uses
`trunc_normal_(pos_embed, std=.02)` and `normal_(cls_token, std=1e-6)` (i.e. the CLS token starts
*almost exactly zero*, not at the same 0.02 scale as everything else) — small, deliberate details
that matter more than they look; blindly using `torch.randn()` (unit variance, as in the
`lucidrains` reference `self.pos_embedding = nn.Parameter(torch.randn(...))`) is a plausible but
non-default alternative — `torch.randn` is std=1, ~50× larger than the timm convention, which can
make the position signal dominate the patch-content signal early in training. **Recommendation: use
`trunc_normal_(std=0.02)` for all Linear/pos-embed weights, zero bias, and zero-init (or
near-zero) the final classification head** — this combination is both the most standard choice and
gives the cleanest, most diagnosable loss-at-init behavior.

---

## 6. Training recipe from scratch

Three primary-source recipes, concrete numbers:

### DeiT (Touvron et al. 2020, arXiv:2012.12877, Table 9)
| | |
|---|---|
| Optimizer | AdamW |
| LR | `0.0005 × batch_size / 512` |
| Batch size | 1024 |
| Epochs | 300 |
| Weight decay | 0.05 |
| Warmup | 5 epochs |
| LR schedule | cosine decay |
| Label smoothing | ε = 0.1 |
| Stochastic depth | 0.1 |
| Dropout | none |
| RandAugment | magnitude 9, mstd 0.5 |
| Mixup | prob 0.8 |
| CutMix | prob 1.0 |
| Random erasing | prob 0.25 |
| Repeated augmentation | yes (3 repetitions) |
Plus DeiT's headline contribution, the **distillation token**: an extra learnable token
(alongside CLS) that attends through the same self-attention layers but is trained to reproduce a
teacher model's predicted label (hard or soft) rather than the ground truth — giving the student a
second, teacher-derived supervisory signal end-to-end via attention rather than only via loss
averaging. **EMA of weights**: the DeiT official codebase supports a `--model-ema` flag; third-party
replications report DeiT-S 300-epoch top-1 improving from 80.0% (AdamW) to roughly 80.2% with EMA
(UNVERIFIED — sourced from a secondary aggregation, not read directly from the DeiT paper's own
ablation table; treat as directional, not exact).

### DeiT III (Touvron et al. 2022, arXiv:2204.07118)
| | ImageNet-1k | ImageNet-21k |
|---|---|---|
| Batch size | 2048 | 2048 |
| Optimizer | LAMB | LAMB |
| LR | 3e-3 | 3e-3 pretrain / 3e-4 finetune |
| Weight decay | 0.02 | 0.02 |
| Warmup | 5 epochs | 5 epochs |
| Epochs | 400 | 90 pretrain + 50 finetune |
| Loss | **Binary cross-entropy** (not softmax CE) | Cross-entropy |
| LayerScale init | 1e-4 (uniform across model sizes) | |
| Stochastic depth | ViT-B 0.1, ViT-L 0.4, ViT-H 0.5 | |
| Train resolution | 160×160 or 192×192, finetuned to 224×224/384×384 | |
| Augmentation | **"3-Augment"**: pick uniformly among {Grayscale, Solarization, Gaussian blur} + horizontal flip + ColorJitter(0.3) | |
Notably, DeiT III trains **single-label ImageNet classification with BCE loss instead of softmax
CE** — directly analogous to how we'll train our genuinely multi-label 12-target problem, i.e. this
recipe family is already validated as compatible with a sigmoid/BCE head. "3-Augment" is explicitly
motivated as *"closer to the practice in self-supervised learning"* — i.e. weaker/simpler than
RandAugment/AutoAugment, deliberately, and reported to work at least as well at these longer epoch
budgets.

### "Better plain ViT baselines for ImageNet-1k" (Beyer, Zhai, Kolesnikov 2022, arXiv:2205.01580)
| | |
|---|---|
| Pooling | GAP (not CLS) — see §3 |
| Pos-embed | fixed 2D sincos — see §2 |
| Optimizer | Adam (`scale_by_adam`), bfloat16 |
| LR | 0.001 |
| Weight decay | 0.0001 |
| Warmup | 10,000 steps |
| Schedule | cosine decay |
| Batch size | 1024 |
| RandAugment | level 10 |
| Mixup | prob 0.2 |
| Dropout / stochastic depth | **none** — *"standard data augmentation is sufficient"* |
| Result | 76.5% top-1 @ 90 epochs (<7h on TPUv3-8); 80.0% @ 300 epochs (<1 day) |
This is the most *deliberately minimal, reproducible* recipe of the three and is a strong template
for a from-scratch implementation: fewer moving parts (no CLS, no distillation, no stochastic depth)
while still reaching competitive accuracy, with a config explicitly designed to be fast to train.

### "How to train your ViT" (Steiner et al. 2021, arXiv:2106.10270)
Defines **AugReg** = {Mixup (α up to 0.8), RandAugment (magnitude/layers), dropout (p=0.1),
stochastic depth (max rate 0.1)} as a bundle, and studies its interaction with dataset size and
compute. Central finding: *"increased compute and AugReg can yield models with the same performance
as models trained on an order of magnitude more training data."* Crucially, the effect is **not
uniform across data regimes** — on **ImageNet-1k (medium, ~1.3M images)**, *"any kind of AugReg
helps"*; on **ImageNet-21k (large, ~13M images)** with fixed compute, *"any kind of AugReg hurts
performance for all but the largest models."* For **very small datasets** (their Pets37 example,
~3,000 images), their conclusion is blunt: don't train from scratch — *"transferring a pre-trained
model is simultaneously significantly cheaper and gives better results"* than extensive
hyperparameter search on the small set from scratch. **This is a direct, primary-source warning
relevant to our 58-verified-label subset**: at *that* scale, AugReg/from-scratch training is
explicitly the wrong regime per this paper — reinforcing why CLAUDE.md's plan already routes around
it (pseudo-labeling to enlarge the effective training set to thousands, not training on 58 examples
directly).

### Cross-cutting practical items (not all directly evidenced above, flagged where so)
- **AdamW betas**: none of the fetched primary sources' extracted text gave an explicit β1/β2 other
  than PyTorch/DeiT defaults; **UNVERIFIED** exact values — PyTorch/AdamW default `(0.9, 0.999)` is
  the near-universal community default for ViT training and is safe to assume absent contrary
  evidence.
- **Gradient clipping**: ViT/DeiT-family training commonly clips gradient norm (commonly max-norm
  1.0 in many public configs); exact DeiT value **UNVERIFIED** from primary sources fetched here —
  verify against the DeiT/`timm` training script (`train.py --clip-grad`) before relying on a
  specific number.
- **Mixed precision, bf16 vs fp16**: "Better plain ViT baselines" trains in **bfloat16**
  (arXiv:2205.01580, "scale_by_adam ... bfloat16"). bf16 has the same exponent range as fp32 (no
  need for loss scaling) at the cost of fewer mantissa bits than fp16; fp16 needs
  `torch.cuda.amp.GradScaler`-style loss scaling to avoid underflow. **The RTX 5070 supports bf16
  natively; prefer bf16 autocast over fp16 to skip the loss-scaling complexity**, consistent with
  the big_vision recipe. (General PyTorch AMP behavior, not paper-specific — flagged as general
  knowledge, not a citation-backed claim.)
- **Gradient checkpointing / accumulation**: not covered by the fetched papers (an engineering, not
  a research, technique) — standard PyTorch `torch.utils.checkpoint` per-block, and accumulating
  gradients over N micro-batches before `optimizer.step()`, are the standard ways to trade compute
  for the 12 GB VRAM ceiling; not paper-sourced, flagged as general engineering practice.

---

## 7. Why ViTs are data-hungry, and remedies for small data

**Root cause (primary source).** The original ViT paper states plainly: *"When trained on mid-sized
datasets such as ImageNet without strong regularization, these models yield modest accuracies of a
few percentage points below ResNets of comparable size"* because Transformers lack the
*"inductive biases inherent to CNNs, such as translation equivariance and locality"*
(arXiv:2010.11929). Concretely: a CNN's conv kernel structurally assumes that nearby pixels are more
related than far pixels and that the same feature detector is useful everywhere in the image; ViT's
self-attention has no such built-in assumption — it has to *learn* locality and translation structure
from data, which needs either a lot of data or explicit help.

**Remedies, in order of "how much they help small-data training," per the sources gathered:**

1. **Convolutional/hybrid stems** — see §1 (Xiao et al. 2021). Re-introduces locality at the input
   only; cheap, broadly applicable, minimal architectural change.
2. **Shifted Patch Tokenization (SPT) + Locality Self-Attention (LSA)** — "Vision Transformer for
   Small-Size Datasets" (Lee, Lee, Song 2021, arXiv:2112.13492,
   https://ar5iv.labs.arxiv.org/html/2112.13492). The paper's motivating observation: standard ViT's
   effective receptive field is *"approximately 30 times smaller than that of the ResNet50
   feature"*. **SPT**: shift the input image diagonally in 4 directions by half a patch width,
   concatenate the shifted copies with the original before patchifying — widens each token's
   effective receptive field before any attention happens. **LSA**: (a) make the softmax temperature
   a **learned** parameter (found to converge to *lower* values than the fixed default, i.e.
   sharper/more selective attention), and (b) **diagonal masking** — exclude a token's attention to
   itself, forcing it to gather information from *other* tokens rather than trivially "attending to
   itself" (which is otherwise a common early-training degenerate solution). Reported gains from
   adding both SPT+LSA on top of existing architectures on **Tiny-ImageNet**: ViT +4.00pp
   (57.07%→61.07%), Swin +4.08pp (60.87%→64.95%), CaiT +2.81pp (64.37%→67.18%); on **CIFAR-100**:
   ViT +3.11pp, PiT +4.01pp; average reported gain across tested ViT variants on Tiny-ImageNet
   ≈ **+2.96pp**. This is a drop-in, architecture-agnostic small-dataset booster and directly
   targeted at our exact problem (training a ViT variant on a comparatively small dataset from
   scratch).
3. **Compact Convolutional Transformers (CCT)** — "Escaping the Big Data Paradigm with Compact
   Transformers" (Hassani et al. 2021, arXiv:2104.05704,
   https://ar5iv.labs.arxiv.org/html/2104.05704). Two changes: (a) a lightweight **convolutional
   tokenizer** (conv + ReLU + maxpool, replacing patchify) that also relaxes ViT's "image size must
   be divisible by patch size" constraint and shrinks token count via pooling (helping the O(N²)
   attention cost); (b) **SeqPool** — replace the CLS token with a learned softmax-weighted pooling
   over the *entire* output token sequence (a lightweight attention-pool, no extra CLS token, no
   extra parameters vs. adding a class token). Reported results: **98% CIFAR-10 accuracy with only
   3.7M params**; reaches 90% CIFAR-10 accuracy in **under 30 minutes on a single RTX 2080 Ti**
   (and even 80% on a CPU in 30 minutes); a 3.85M-param CCT-7/3×2 hits 95.04% on CIFAR-10 at 300
   epochs, described as *"15% the size of ResNet50"* with comparable accuracy. This is the strongest
   evidence in the literature that a *from-scratch*, small transformer can be trained end-to-end,
   fast, on genuinely small data, provided the tokenizer and pooling are convolution/attention-pool
   based rather than plain-patchify + CLS.
4. **Heavy augmentation** — RandAugment, Mixup, CutMix, random erasing (DeiT, §6); Steiner et al.'s
   AugReg bundle (§6) shows augmentation/regularization substitutes for roughly an order of magnitude
   of missing training data *in the ImageNet-1k-sized regime*, but note their explicit warning that
   at Pets37-scale (~3k images, closer to our 58-verified-label set) this stops being sufficient and
   transfer/pretraining wins outright (arXiv:2106.10270).
5. **Distillation** — DeiT's distillation token (§6) lets a data-efficient ViT borrow a CNN teacher's
   inductive bias indirectly through soft/hard label supervision, without architectural changes to
   the student beyond the extra token.
6. **Self-supervised pretraining** — mentioned only for completeness; MAE (§2) and MoCo v3 style
   pretraining reduce the *labeled*-data requirement by first learning representations on unlabeled
   images (relevant since we have far more unlabeled/report-only studies than verified-labeled
   ones), but this is covered in depth by a different research stream per the task split — flagged
   here only as an existing, validated remedy category, not expanded on.

**Bottom line for our situation:** with ~7,500 studies total but only 58 verified labels (per
CLAUDE.md), the "How to train your ViT" finding that AugReg cannot substitute for real data at
Pets37-like scale (§6) argues that pseudo-labeling to expand the *effective* labeled set is not
optional polish — it's the difference between a data regime where ViT training from scratch is
known to work reasonably (ImageNet-1k-like, low thousands+ of examples with augmentation) and one
where the primary literature says it doesn't (a few-thousand-or-fewer regime without heavy structural
help). Architecturally, SPT+LSA and/or a convolutional tokenizer (CCT-style) are the most
directly-evidenced, low-risk small-data boosters to combine with whatever data-scale we land on.

---

## 8. Model size vs. dataset size guidance

**Standard ViT model configs** (arXiv:2010.11929, Table 1):

| Model | Layers | Hidden size D | MLP size | Heads | Params |
|---|---|---|---|---|---|
| ViT-Base | 12 | 768 | 3072 | 12 | ~86M |
| ViT-Large | 24 | 1024 | 4096 | 16 | ~307M |
| ViT-Huge | 32 | 1280 | 5120 | 16 | ~632M |

The commonly-used smaller **DeiT-family** sizes (ViT-Tiny/Small), introduced alongside DeiT and used
throughout `timm`, follow the same architecture with fewer parameters (Tiny ≈5M, Small ≈22M params —
**UNVERIFIED exact figures**, not read directly from a primary source in this research pass; these
are the widely-cited `timm`/DeiT community numbers for `deit_tiny_patch16_224` /
`deit_small_patch16_224` and should be confirmed directly from `timm`'s model registry before being
relied on for a paper/writeup).

**Guidance for ~7,500 studies / hundreds of thousands of 2D slices:** none of the fetched papers give
a direct "N images per M parameters" rule; the closest primary evidence is Steiner et al.'s AugReg
study (§6/§7) showing that even at **ImageNet-1k scale (1.3M images)**, "any kind of AugReg helps,"
implying that scale is still on the data-hungry side of the curve for full-size ViT-Base, and that at
**Pets37 scale (~3k images)** the recommendation flips to "don't train from scratch, use a pretrained
model." Our situation — even optimistically counting hundreds of thousands of 2D slices — sits
architecturally between these two references depending on how we tokenize (whole-study/3D vs.
per-slice/2D), and more importantly, our effective **distinct-study** count (thousands, not
millions) matters more than raw slice count, since slices within a study are highly correlated, not
independent training signal. **Practically (this is a derived recommendation, not a literature
number): start at ViT-Tiny/Small scale, not ViT-Base**, given (a) 12 GB VRAM caps batch size and
token count together, (b) the data-hungriness evidence above argues against expecting a full
ViT-Base to train well from scratch on a dataset several orders of magnitude smaller than the
ImageNet-1k regime where AugReg was shown to be sufficient, and (c) `CLAUDE.md`'s own fallback plan
(switch to a pretrained backbone if the scratch ViT isn't beating baseline by the Oct 8 decision
date) is the correct safety valve if Tiny/Small underperforms — there's little value in burning
VRAM/compute budget on a Base-scale model whose main literature-backed prerequisite (order of
magnitude more data than we have, or heavy AugReg at that data scale) isn't met.

---

## 9. Debugging a from-scratch transformer

Concrete checklist (primary source: Karpathy, "A Recipe for Training Neural Networks," 2019,
https://karpathy.github.io/2019/04/25/recipe/, supplemented with ViT-specific detail):

1. **Loss-at-init sanity check.** With a zero-initialized head (§5), our 12-way sigmoid/BCE head
   should output logit 0 for every class at step 0, giving `BCEWithLogitsLoss ≈ -log(0.5) = 0.6931`
   per label, averaged over 12 labels ≈ **0.693** total. Karpathy's version of this check
   (softmax case): *"you should measure `-log(1/n_classes)` on a softmax at initialization"* — the
   same idea applied to our sigmoid/multi-label setup. If the measured init loss is *far* from
   0.693, something is wrong before you've spent any real training time (e.g. head not actually
   zero-initialized, a stray non-zero bias, wrong loss reduction, or label tensor dtype/shape bug).
2. **Overfit one batch.** Take a tiny batch (a handful of studies) and train only on it, watching
   loss go to (near) zero and predictions match exactly — Karpathy: *"Overfit a single batch of only
   a few examples... increase model capacity until you achieve near-zero loss."* This is the
   single highest-value first test of a from-scratch training loop — it catches most "the model
   can't learn at all" bugs (see common bugs below) before you touch the real dataset.
3. **Input visualization immediately before `model(x)`.** Karpathy: *"The unambiguously correct
   place to visualize your data is immediately before your `y_hat = model(x)`"* — for us this means
   visually inspecting the exact normalized/resampled tensor the model receives (post-DICOM
   preprocessing, post-augmentation), not the raw DICOM — preprocessing bugs (wrong slice order,
   wrong window/level, accidental double-normalization) are otherwise invisible.
4. **Attention map visualization.** Not covered by the fetched Karpathy source directly, but
   standard ViT practice (used throughout the ViT/CaiT/LSA papers' own figures, e.g. LSA's Figure 3
   in arXiv:2112.13492 visualizing learned temperature/attention patterns): plot a head's attention
   weights from the CLS/pooling query back onto the image grid. A model that's learning something
   sensible should show attention concentrating on plausible anatomical regions over training, not
   uniform or degenerate (all weight on one token, e.g. itself — exactly the failure mode LSA's
   diagonal masking is designed to prevent, §7).
5. **Gradient norm monitoring.** Track the global gradient norm (and per-block, if debugging
   instability) over training. Sudden spikes/NaNs are the practical symptom of the unbounded
   attention-logit growth ViT-22B's QK-norm paper documents at scale (§4, arXiv:2302.05442,
   *"attention logits grew to over 50,000 in magnitude"* without normalization) — even though that
   paper's regime is much larger than ours, the diagnostic (watch gradient/logit norms, add QK-norm
   or gradient clipping if they blow up) transfers directly.
6. **Common from-scratch ViT bugs** (compiled from the code read in this research pass, i.e.
   comparing `timm`'s and `lucidrains`' reference implementations against likely first-draft
   mistakes):
   - **Wrong reshape when splitting heads.** The standard correct pattern is
     `qkv.reshape(B, N, 3, num_heads, head_dim).permute(...)` or, per `lucidrains`,
     `rearrange(t, 'b n (h d) -> b h n d', h=heads)` — a common bug is reshaping `(h d)` as `(d h)`
     (i.e. getting the head/head-dim split order backwards), which silently produces a
     dimensionally-correct but semantically-scrambled multi-head split (heads mix channels that
     were never meant to be grouped) — this trains, sometimes even converges, but underperforms in a
     way that's hard to catch without an explicit shape/identity test.
   - **Missing the `1/√d` attention scale.** Both reference implementations explicitly apply
     `self.scale = head_dim ** -0.5` before/with the `q @ k.T` product (`lucidrains`:
     `dots = q @ k.T * self.scale`; `timm`: `q = q * self.scale` before the manual path, or passed
     implicitly inside `F.scaled_dot_product_attention`, which applies `1/√d` internally by
     default). Omitting this scale makes the pre-softmax logits grow with `head_dim`, pushing
     softmax toward a near-one-hot / vanishing-gradient regime, especially at larger head dims.
   - **Positional embedding not actually added / added to the wrong tensor** (e.g. added before
     the CLS token is concatenated, so the CLS token gets no positional information, or added after
     a LayerNorm that then gets undone) — both reference implementations add positional embeddings
     to the *full* sequence (patches + CLS) right after concatenation and before the first
     transformer block; verify with a simple `assert not torch.equal(x_with_pos, x_without_pos)`-
     style check during development.
   - **LayerNorm placement (pre- vs post-norm) getting mixed mid-implementation** — e.g.
     accidentally normalizing the residual *stream* itself in-place, rather than normalizing only
     the sublayer's *input* while leaving the residual path un-normalized. The correct pre-norm
     residual is `x = x + Sublayer(LN(x))`, not `x = LN(x + Sublayer(x))` (post-norm) and definitely
     not `x = LN(x) + Sublayer(LN(x))` (double-normalizes the skip connection, which is not the
     'pre-norm' the literature refers to at all).

---

## 10. Multi-label heads (12 labels)

**Baseline: 12 independent sigmoid logits + `BCEWithLogitsLoss`.** Standard, correct formulation for
"which of these 12 independent binary findings are present" (as opposed to single-label softmax,
which is for *mutually exclusive* classes) — this matches DeiT III's own use of BCE loss even for
single-label ImageNet (§6, arXiv:2204.07118), underscoring that BCE-per-class is a validated,
standard ViT head design, not an exotic choice.

**`pos_weight`.** `BCEWithLogitsLoss(pos_weight=...)` up-weights the positive class per-label to
counteract label imbalance (e.g. Fracture/Contusion are presumably much rarer than Effusion or
Medial Meniscus tear) — a per-label scalar, typically `num_negatives/num_positives` computed from
the training set for each of the 12 labels independently, since prevalence differs a lot per
finding (per `CLAUDE.md`'s note that "prevalence may differ between train / public LB / private
LB" and each finding is macro-averaged equally in the AUC metric regardless of prevalence — meaning
rare-label learning signal matters just as much as common-label signal to the final score).

**Asymmetric Loss (ASL)** (Ben-Baruch et al. 2020, arXiv:2009.14119,
https://ar5iv.labs.arxiv.org/html/2009.14119) generalizes focal-loss-style down-weighting
asymmetrically for multi-label problems: separate focusing exponents for positives/negatives
(`L₊ = (1-p)^γ₊ log(p)`, `L₋ = p^γ₋ log(1-p)` with `γ₋ > γ₊` so **easy negatives are down-weighted
much more aggressively than positives**), plus a **hard-negative probability margin `m`**
(`p_m = max(p - m, 0)`, applied inside the negative-loss term) that *completely discards very easy
negatives* from the gradient — a mechanism plain BCE (and plain focal loss, which only has one
shared γ) doesn't have. Reported gains over BCE: **MS-COCO 86.6% mAP (ASL) vs 84.0% (BCE)**
(+2.6pp), **Open Images 86.3% vs 84.8% micro-mAP**. This is directly relevant to our label-imbalance
situation (12 findings likely with very different prevalence) and is a natural upgrade path once a
plain BCE baseline is working: it's a drop-in loss-function swap, no architecture change required.

**Label-specific attention queries (ML-Decoder / Query2Label, detailed in §3).** Instead of pooling
to one global feature vector and running all 12 logits off it, give each of the 12 labels its own
learned query that cross-attends into the token sequence, producing a per-label pooled feature before
its own linear→sigmoid. This is the head-architecture-level analogue of what ASL/pos_weight do at the
loss level (both are about giving each label a fairer, more tailored shot at good gradients) — the
two are complementary, not competing: **a reasonable end-state combines a 12-query cross-attention
pooling head (§3) with an ASL or pos-weighted BCE loss (this section).**

**Recommended starting point, in order of implementation cost:** (1) GAP + linear(D→12) +
`BCEWithLogitsLoss(pos_weight=...)` — get this working and validated first (cheapest, matches §3's
"start with GAP" recommendation); (2) swap `pos_weight`-BCE for **ASL** once label-imbalance is
empirically confirmed to be a bottleneck (check per-label AUC, not just aggregate, since macro-AUC
weights every label equally per `CLAUDE.md`'s metric definition); (3) upgrade the pooling head to
**12 learned cross-attention queries** (Query2Label-style) as an architecture ablation once the
simpler pipeline is validated end-to-end.

---

## Recommended default design for our from-scratch ViT

Given: ~7,500 studies (58 verified-labeled; rest pseudo/report-labeled), 12 GB VRAM, 16 GB RAM,
12 independent binary targets scored by macro-AUC, RTX 5070 (bf16-capable).

1. **Tokenizer**: small convolutional stem (Xiao et al. §1) or CCT-style conv tokenizer (§7) instead
   of a single patchify Conv2d — cheap locality bias, improves optimizer robustness and
   small-data convergence.
2. **Positional encoding**: fixed 2D sin-cos (§2, Beyer et al.) as the default; keep RoPE-mixed
   (§2, RoPE-ViT) on the ablation list, especially useful if we later need to generalize across
   varying slice/resolution grids.
3. **Pooling**: start with GAP (§3, Beyer et al.'s direct GAP > CLS result); upgrade path is a
   12-query cross-attention pooling head, one query per finding (§3, §10).
4. **Block**: pre-norm, LayerNorm `eps=1e-6`, MHSA via `F.scaled_dot_product_attention` (§4),
   MLP ratio 4 with GELU, LayerScale (`init_values≈0.1` at our depth per CaiT's schedule, §4),
   linear stochastic-depth schedule to a modest max rate (start low, e.g. 0.1, given our smaller
   model/data scale vs. DeiT III's larger-model 0.4–0.5), dropout optional/likely unnecessary if
   augmentation is used (per Beyer et al., §4/§6). Consider QK-norm (§4) as a cheap stabilizer,
   especially under bf16/fp16.
5. **Small-data structural help**: add SPT + LSA (§7, Lee et al.) — diagonal-masked, learned-
   temperature attention plus shifted-patch tokenization — directly evidenced to help exactly this
   regime (+~3pp average on small-dataset benchmarks), and complements the conv stem rather than
   duplicating it.
6. **Initialization**: `trunc_normal_(std=0.02)` for Linear/pos-embed weights, zero bias; **zero-init
   the classification head** (weight and bias) so BCE loss at init ≈ 0.693/label (§5, §9 sanity
   check); near-zero init (`std=1e-6`) for any CLS/query tokens if used.
7. **Model size**: start at ViT-Tiny/Small scale, not Base (§8) — matches VRAM budget and the
   data-hungriness evidence that Base-scale ViTs need far more data than we have without heavy
   structural/augmentation help.
8. **Training recipe**: AdamW, bf16 autocast (native on RTX 5070, avoids fp16 loss-scaling, matches
   the big_vision/Beyer-et-al. recipe, §6), cosine LR decay with warmup, moderate RandAugment +
   Mixup (DeiT/Steiner-et-al.-style, tuned down given our smaller data/model scale — AugReg helps at
   ImageNet-1k scale but Steiner et al. explicitly warn it stops helping at very small (~3k-image)
   scale, §6/§7, which argues for *not* over-relying on it if our effective labeled set stays small;
   revisit once pseudo-labeling has actually expanded the effective training set).
   Gradient checkpointing + accumulation to fit the 12 GB ceiling; gradient-norm monitoring during
   training (§9).
9. **Head/loss**: 12 sigmoid logits, start with `BCEWithLogitsLoss(pos_weight=...)` per-label,
   upgrade to Asymmetric Loss (§10) once per-label AUC shows an imbalance bottleneck.
10. **Debugging discipline before any real training run**: verify loss-at-init ≈ 0.693 (§9),
    overfit one small batch to ~0 loss (§9), visualize the exact tensor entering `model(x)` (§9,
    catches DICOM-preprocessing bugs specifically), and periodically visualize attention maps once
    training is underway.

Everything above is an ablation-worthy default, not a fixed spec — the CLAUDE.md Oct 8 decision date
(switch to a pretrained-backbone fallback if the scratch ViT isn't clearly beating a simple baseline)
is the right forcing function to stop tuning this recipe and ship, if it's not working by then.

---

## Sources

- ViT (original): Dosovitskiy et al. 2020, "An Image is Worth 16x16 Words," arXiv:2010.11929 — https://arxiv.org/abs/2010.11929 (full text via https://ar5iv.labs.arxiv.org/html/2010.11929)
- DeiT: Touvron et al. 2020, "Training data-efficient image transformers & distillation through attention," arXiv:2012.12877 — https://ar5iv.labs.arxiv.org/html/2012.12877
- DeiT III: Touvron et al. 2022, "DeiT III: Revenge of the ViT," arXiv:2204.07118 — https://ar5iv.labs.arxiv.org/html/2204.07118
- Better plain ViT baselines: Beyer, Zhai, Kolesnikov 2022, arXiv:2205.01580 — https://ar5iv.labs.arxiv.org/html/2205.01580
- How to train your ViT: Steiner et al. 2021, arXiv:2106.10270 — https://ar5iv.labs.arxiv.org/html/2106.10270
- Early Convolutions Help Transformers See Better: Xiao et al. 2021, arXiv:2106.14881 — https://ar5iv.labs.arxiv.org/html/2106.14881
- MAE: He et al. 2021, "Masked Autoencoders Are Scalable Vision Learners," arXiv:2111.06377 — https://ar5iv.labs.arxiv.org/html/2111.06377
- Compact Convolutional Transformers (CCT): Hassani et al. 2021, "Escaping the Big Data Paradigm with Compact Transformers," arXiv:2104.05704 — https://ar5iv.labs.arxiv.org/html/2104.05704
- Vision Transformer for Small-Size Datasets (SPT+LSA): Lee, Lee, Song 2021, arXiv:2112.13492 — https://ar5iv.labs.arxiv.org/html/2112.13492
- ViT-22B (QK-norm, parallel layers): Dehghani et al. 2023, arXiv:2302.05442 — https://ar5iv.labs.arxiv.org/html/2302.05442
- CaiT (LayerScale): Touvron et al. 2021, "Going Deeper with Image Transformers," arXiv:2103.17239 — https://ar5iv.labs.arxiv.org/html/2103.17239
- RoPE-ViT: Heo et al. 2024, "Rotary Position Embedding for Vision Transformer," ECCV 2024, arXiv:2403.13298 — https://arxiv.org/abs/2403.13298; code: https://github.com/naver-ai/rope-vit
- Asymmetric Loss (ASL): Ben-Baruch et al. 2020, arXiv:2009.14119 — https://ar5iv.labs.arxiv.org/html/2009.14119
- ML-Decoder: Ridnik et al. 2021, arXiv:2111.12933 — https://ar5iv.labs.arxiv.org/html/2111.12933
- Query2Label: Liu et al. 2021, arXiv:2107.10834 — https://ar5iv.labs.arxiv.org/html/2107.10834
- timm `vision_transformer.py` (Ross Wightman et al.), `patch_embed.py`, `attention.py`, `drop.py` — https://github.com/huggingface/pytorch-image-models (files fetched directly from `main` branch, 2026-09-24)
- lucidrains `vit-pytorch`, `vit.py` — https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
- google-research/vision_transformer (official JAX reference, cited by timm's docstrings for JAX-style init and patch-embed design) — https://github.com/google-research/vision_transformer
- Karpathy, "A Recipe for Training Neural Networks," 2019 — https://karpathy.github.io/2019/04/25/recipe/
- DeiT EMA ablation numbers (§6): sourced from a secondary web-search aggregation, not read directly from the DeiT paper's own table — marked UNVERIFIED, verify against arXiv:2012.12877 Table 9 / official repo ablations before relying on it.
- ViT-Tiny/Small exact parameter counts (§8): community/`timm`-registry standard figures, not confirmed from a primary source in this pass — marked UNVERIFIED, verify via `timm.create_model('vit_tiny_patch16_224')` param count before citing.
- AdamW betas and DeiT gradient-clip value (§6): not found explicitly in the fetched primary-source excerpts — marked UNVERIFIED, verify against the DeiT/`timm` training scripts directly.
