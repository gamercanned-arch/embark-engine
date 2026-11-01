# DiT Algorithm & Diffusion Review: FLUX.2 Klein 4B Megakernel Plan (v1.0)

**Critic:** Critic 2 — DiT Algorithm & Diffusion Specialist  
**Target Plan:** `c:\Users\abhik\Documents\antigravity\silly-pascal\scratch\flux2_dit_megakernel_plan_v1.md`  
**System Target:** AMD Ryzen 7 7730U APU (Zen 3 Barcelo-R, Vega 8 `gfx90c`, 16 GB UMA DDR4-3200)  
**Workload Target:** FLUX.2 Klein 4B (`Q3_K_M`), Qwen3-4B Text Conditioning, 16-channel VAE, 4-Step Flow-Matching  

---

## 1. Executive Evaluation & Score

### **Overall Score: 4.5 / 10**

**Verdict:** *Algorithmically broken and mathematically inconsistent.*  
While the plan demonstrates good high-level intuition regarding memory staging on a 16 GB UMA APU and correctly identifies the 16-channel latent space and 4-step flow-matching paradigm, its core mathematical formulation of the Diffusion Transformer (DiT) contains critical errors. It misplaces the AdaLN-zero residual gating parameter, applies 2D RoPE indiscriminately across concatenated text-image sequences, conflates text token count with feature dimension, omits the mandatory resolution-dependent flow-matching time-shift schedule, and claims L3 cache residency based on an activation calculation that is off by an order of magnitude.

If implemented as written, the kernel will either produce numerical NaN/divergence or generate incoherent high-frequency noise.

---

## 2. Top 3 Strengths

1. **Strict Sequential UMA Lifecycle Staging:**
   The architectural decision in Section 1 to enforce disjoint lifetimes for Qwen3-4B ($\approx 2.36\text{ GB}$), FLUX.2 Klein DiT ($\approx 2.12\text{ GB}$), and the Tiled VAE ($\approx 336\text{ MB}$) is essential for the 16 GB unified memory architecture of the Ryzen 7 7730U. Preventing working-set overlap holds peak commit charge under 4 GB, avoiding OS paging stalls.

2. **Accurate Recognition of Latent Topology & 4-Step Paradigm:**
   The plan correctly identifies the 16-channel latent structure ($8\times$ spatial downsampling) characteristic of the FLUX autoencoder and targets a 4-step distilled flow-matching trajectory, which is the mathematically appropriate operating regime for real-time edge DiT inference.

3. **Tiled VAE Mitigation Strategy:**
   Acknowledging memory pressure during the latent decoding stage and proposing spatial tiling with blending overlap prevents activation spikes during the unrolling of the high-channel convolutional decoder resblocks.

---

## 3. Critical Mathematical & Algorithmic Flaws

### Flaw 1: AdaLN-Zero Inversion & Residual Gating Misplacement
* **Plan Statement (Section 2.B.2):**  
  > *"AdaLN-zero In-Flight Scaling: Scale and shift vectors ($(\gamma, \beta, \alpha)$) computed in registers and applied directly to normalized tokens, avoiding extra memory roundtrips."*
* **Mathematical Reality & Consequence:**  
  In DiT (Peebles & Xie) and FLUX architectures, Adaptive Layer Normalization with Zero-Initialization (AdaLN-Zero) produces **three distinct vectors per sub-layer** (Attention and MLP):
  1. Affine scale ($\gamma$) and shift ($\beta$) for pre-normalization:
     $$\hat{x} = \text{LayerNorm}(x) \odot (1 + \gamma) + \beta$$
  2. Residual gate ($\alpha$) for the post-layer skip connection:
     $$x_{\text{next}} = x + \alpha \odot \text{SubLayer}(\hat{x})$$
  
  The plan proposes applying $(\gamma, \beta, \alpha)$ *simultaneously to the normalized tokens before the sublayer*. This is fatal:
  - If $\alpha$ is multiplied prior to projection, the residual skip connection is unscaled ($x + \text{SubLayer}(\dots)$).
  - In AdaLN-Zero, $\alpha$ is initialized/trained near 0 so that blocks initially behave as identity transforms. Conflating the residual gate with input modulation removes identity conditioning, immediately triggering gradient/activation explosion.
  - Furthermore, in FLUX Double-Stream blocks, the image and text streams possess **independent modulation pathways**, producing 12 modulation parameters per block ($[\gamma_{1x}, \beta_{1x}, \alpha_{1x}, \gamma_{2x}, \beta_{2x}, \alpha_{2x}]$ for image, and $[\gamma_{1y}, \beta_{1y}, \alpha_{1y}, \gamma_{2y}, \beta_{2y}, \alpha_{2y}]$ for text). In Single-Stream blocks, image and text streams share unified linear projections with a unified modulation. The plan fails to differentiate between these two stream topologies.

---

### Flaw 2: 2D RoPE Coordinate Assignment on Joint Streams & Wasted Runtime Trigonometry
* **Plan Statement (Section 2.B.1):**  
  > *"2D RoPE Rotary Embeddings: Dual-axis frequency modulation computed directly in AVX2 YMM registers using fast polynomial approximations for `sin`/`cos`."*
* **Mathematical Reality & Consequence:**  
  1. **Coordinate Misalignment in Concatenated Joint Streams:**  
     FLUX concatenates text tokens ($L_{\text{txt}} = 512$) and image tokens ($L_{\text{img}} = 2304$ for $768\times 768$ at $2\times 2$ patch size) for joint attention. Image tokens carry 2D spatial grid coordinates $(h_i, w_i) \in [0, 47] \times [0, 47]$. Text tokens do not belong to the 2D spatial grid.  
     FLUX addresses this via **3D/4D RoPE coordinate assignment**:
     $$\text{Pos}_{\text{txt}}[i] = (i, 0, 0), \quad \text{Pos}_{\text{img}}[j] = (0, h_j, w_j)$$
     The rotary embedding splits each attention head ($d_{\text{head}} = 128$) into frequency bands dedicated to text index $t$, row index $h$, and column index $w$. Applying a generic "dual-axis 2D RoPE" across the concatenated sequence without coordinate masking will rotate text tokens with spatial frequencies and corrupt text-image cross-attention alignment.
  2. **Wasted Compute on Polynomial Approximations:**  
     For a fixed resolution ($768\times 768$) and fixed prompt padding ($L_{\text{txt}} = 512$), RoPE frequencies $\cos(\omega_k p)$ and $\sin(\omega_k p)$ are **100% invariant across all 4 denoising steps and all 38+ transformer layers**.  
     The entire table for $(512 + 2304) = 2816$ tokens $\times 64$ frequency pairs requires only:
     $$2816 \times 64 \times 2 \ (\cos, \sin) \times 2\text{ bytes (FP16)} \approx 704\text{ KiB}$$
     Evaluating polynomial trigonometric approximations dynamically in AVX2 registers across 38 blocks $\times$ 4 steps burns tens of millions of ALU cycles for zero mathematical benefit and introduces numerical precision drift.

---

### Flaw 3: Flow-Matching Euler Integration: Missing Resolution Time-Shift & Undefined Guidance Schedule
* **Plan Statement (Section 1 & 2):**  
  > *"4 Flow-Matching Denoising Steps (Vega 8 + Zen 3 AVX2)"*
* **Mathematical Reality & Consequence:**  
  1. **Absence of Resolution-Dependent Time Shifting:**  
     In Flow Matching ODEs ($dx_t = v_\theta(x_t, t)dt$), Euler integration with 4 steps cannot use a naive linear schedule $t \in [1.0, 0.75, 0.50, 0.25]$. FLUX requires an empirical non-linear time shift $\sigma(t)$ based on the image token sequence length $N = H \cdot W / 4$:
     $$t_{\text{shifted}} = \frac{s \cdot t}{1 + (s - 1) \cdot t}, \quad \text{where } s = \mu(N) = a + b \cdot \sqrt{N}$$
     For $768\times 768$ ($N=2304$), $s \approx 0.85 - 1.15$. Skipping this transformation shifts the vector field evaluation away from the trained trajectories, causing severe contrast collapse, noise graininess, and incomplete denoising.
  2. **Undefined Classifier-Free Guidance (CFG) vs Guidance Scale Embedding:**  
     The plan does not state whether FLUX.2 Klein 4B uses distilled guidance embeddings or standard dual-pass CFG.
     - If true CFG is required ($v = v_{\text{uncond}} + w \cdot (v_{\text{cond}} - v_{\text{uncond}})$), the number of DiT forward passes doubles from 4 to 8, completely invalidating the latency and bandwidth budget.
     - If distilled guidance is used (as in FLUX.1 Dev), the guidance scale $w$ must be passed through a dedicated guidance embedding MLP and summed with the timestep sinusoidal embedding: $c_{\text{time}} = \text{MLP}_t(\text{embed}(t)) + \text{MLP}_g(\text{embed}(w))$. The plan contains no guidance embedding pathway.

---

### Flaw 4: Token Patching, Activation Expansion, and the 16 MB L3-Residency Mathematical Impossibility
* **Plan Statement (Section 2.B.4):**  
  > *"The active $16 \times 96 \times 96$ latent tensor (147 KB in FP16) and attention context buffers remain 100% resident in L3 cache."*
* **Mathematical Reality & Consequence:**  
  1. **Arithmetic Error:**  
     $16 \times 96 \times 96 \times 2\text{ bytes (FP16)} = 294,912\text{ bytes} \approx 288\text{ KiB}$ (the plan claims 147 KB, undercounting by $2\times$).
  2. **Neglect of Patch Embedding & Hidden Dimension Expansion:**  
     DiT does not process the $16\times 96\times 96$ latent directly. FLUX applies a $2\times 2$ patch packing:
     $$(16, 96, 96) \xrightarrow{\text{pack}} (48 \times 48, 16 \times 2 \times 2) = (2304, 64)$$
     These 2304 tokens are projected via `img_in` to $d_{\text{model}} = 3072$.
     The active image activation tensor alone is:
     $$2304 \times 3072 \times 2\text{ bytes} = 14,155,776\text{ bytes} \approx 13.5\text{ MiB}$$
     Adding the 512 text tokens ($512 \times 3072 \times 2\text{ bytes} \approx 3.0\text{ MiB}$), the base token state is **$16.5\text{ MiB}$**.
     This already exceeds the entire 16 MB L3 cache!
     Furthermore, computing Multi-Head Attention requires Q, K, V projections ($3 \times 16.5\text{ MiB} \approx 49.5\text{ MiB}$), and the MLP intermediate expands tokens to $d_{\text{ff}} = 12288$ ($2304 \times 12288 \times 2\text{ bytes} \approx 54\text{ MiB}$).  
     Claiming that active latents and attention context buffers remain "100% resident in L3 cache" is mathematically false by an order of magnitude.

---

### Flaw 5: Text Conditioning Dimensionality Collapse & Missing Pooled Pathway
* **Plan Statement (Section 1):**  
  > *"Step 1: Text Conditioning (Qwen3-4B) -> Emit 512-dim Prompt Tokens"*
* **Mathematical Reality & Consequence:**  
  - Qwen3-4B does not emit "512-dim tokens". Qwen models feature hidden dimensions $d \in \{2048, 2560, 3072, 3584\}$. The value 512 refers to the **maximum sequence length** ($L_{\text{seq}} = 512$), not the embedding dimensionality!
  - FLUX conditioning requires **two distinct text representations**:
    1. **Sequence Tokens** $Y_{\text{seq}} \in \mathbb{R}^{512 \times d_{\text{text}}}$: projected to $d_{\text{model}} = 3072$ via `txt_in` for cross/joint attention.
    2. **Pooled Context Vector** $y_{\text{pooled}} \in \mathbb{R}^{d_{\text{pool}}}$: pooled across the prompt and projected into the AdaLN conditioning MLP alongside timestep $t$.
  - Without the pooled text vector, the AdaLN-zero modulation receives only timestep information, leaving the global composition and style modulation unconditioned.

---

### Flaw 6: Fine-Grained Heterogeneous Split Synchronization Trap
* **Plan Statement (Section 2.A & 2.B):**  
  QKV/MLP GEMMs on Vega 8 iGPU; RoPE, AdaLN, and Normalization on Zen 3 CPU via zero-copy host buffers.
* **Algorithmic Reality & Consequence:**  
  Within every single transformer block:
  $$\text{CPU (Norm/AdaLN)} \xrightarrow{\text{sync}} \text{GPU (QKV GEMM)} \xrightarrow{\text{sync}} \text{CPU (RoPE)} \xrightarrow{\text{sync}} \text{Attention} \xrightarrow{\text{sync}} \text{GPU (Out Proj)} \xrightarrow{\text{sync}} \text{CPU (MLP AdaLN)} \dots$$
  - This requires **4 to 6 host-device synchronization barriers per block**.
  - Across 38 blocks $\times$ 4 steps = **600 to 900 CPU-GPU synchronization events**.
  - Even with zero-copy UMA (`CL_MEM_USE_HOST_PTR`), cache coherency protocols on x86 APUs require OpenCL memory barriers and CPU cache flushes/invalidations. The driver and barrier overhead will completely swamp kernel execution time, reducing compute efficiency to under 15%.

---

## 4. Specific, Actionable Algorithmic Fixes

### Fix 1: Formulate Exact AdaLN-Zero Modulations for Double & Single Streams

Split the modulation into distinct sub-layer projections:
```cpp
// AdaLN-Zero mathematically correct sublayer formulation:
// 1. Double-Stream Image Block:
// [gamma1_img, beta1_img, alpha1_img, gamma2_img, beta2_img, alpha2_img] = MLP_img(c_vec)
// 2. Pre-Attention Norm & Modulate:
x_norm = LayerNorm(x_img);
x_mod  = x_norm * (1.0f + gamma1_img) + beta1_img;

// 3. Attention Forward Pass:
attn_out = Attention(Q(x_mod), K_joint, V_joint);

// 4. Post-Attention Residual Gating (CRITICAL: alpha scales branch, NOT normalized input):
x_img = x_img + alpha1_img * attn_out;

// 5. Pre-MLP Norm & Modulate:
mlp_norm = LayerNorm(x_img);
mlp_mod  = mlp_norm * (1.0f + gamma2_img) + beta2_img;

// 6. Post-MLP Residual Gating:
x_img = x_img + alpha2_img * MLP(mlp_mod);
```

---

### Fix 2: 3D RoPE Precomputed Look-Up Table (LUT)

Replace dynamic AVX2 polynomial trigonometry with a precomputed static LUT generated once during model load:
- Allocate a contiguous 704 KiB buffer for $(L_{\text{txt}} + L_{\text{img}}) = (512 + 2304) = 2816$ tokens.
- Assign coordinate vectors:
  - For $i \in [0, 511]$: $\vec{p}_i = (i, 0, 0)$
  - For $j = (h \cdot 48 + w) \in [0, 2303]$: $\vec{p}_{512+j} = (0, h, w)$
- Head dimension $d_{\text{head}} = 128$ is split into:
  - $d_t = 32$ channels for text sequence axis
  - $d_h = 48$ channels for vertical image axis
  - $d_w = 48$ channels for horizontal image axis
- Apply RoPE in AVX2 via simple complex multiplication from the precomputed table:
  $$\begin{bmatrix} Q_{\text{rot}, 2k} \\ Q_{\text{rot}, 2k+1} \end{bmatrix} = \begin{bmatrix} \cos & -\sin \\ \sin & \cos \end{bmatrix} \begin{bmatrix} Q_{2k} \\ Q_{2k+1} \end{bmatrix}$$

---

### Fix 3: Implement Resolution Time-Shifted Flow-Matching Schedule

Replace the naive step schedule with the Black Forest Labs shifted schedule:
```python
def get_flux_schedule(num_steps=4, image_seq_len=2304, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15):
    # Compute resolution-dependent shift factor s
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = m * image_seq_len + b
    
    # Linear timesteps from 1.0 (pure noise) to 0.0 (data)
    t_linear = [1.0 - i / num_steps for i in range(num_steps + 1)]
    
    # Apply non-linear time shift
    t_shifted = [(mu * t) / (1.0 + (mu - 1.0) * t) for t in t_linear]
    return t_shifted # Yields exact integration points for Euler steps
```

Euler update at step $k$:
$$x_{k+1} = x_k + (t_{\text{shifted}}[k+1] - t_{\text{shifted}}[k]) \cdot v_\theta(x_k, t_{\text{shifted}}[k])$$

---

### Fix 4: Correct Latent Patching & Block-Level Fusion Architecture

1. **Explicit Patching Layer:**
   - Ingest raw latent $z \in \mathbb{R}^{16 \times 96 \times 96}$.
   - Unfold into non-overlapping $2\times 2$ patches: $z_{\text{patch}} \in \mathbb{R}^{2304 \times 64}$.
   - Linear projection: `img_in(z_patch)` $\to X_{\text{img}} \in \mathbb{R}^{2304 \times 3072}$.
2. **Coarse-Grained Heterogeneous Scheduling (Block-Level Fusion):**
   - Eliminate ping-pong synchronization! Do **not** split sub-layers between CPU and GPU.
   - **Recommended Partition:**
     - **Option A (All-iGPU Pipeline):** Keep entire DiT forward pass (Norm, AdaLN, QKV, Attention, MLP) on Vega 8 using OpenCL fused kernels. CPU only handles text conditioning and overall step dispatch.
     - **Option B (Asynchronous Stream Split):** If CPU assistance is required, assign the entire Text Stream to Zen 3 (8 cores AVX2) and the entire Image Stream to Vega 8 during Double-Stream blocks, synchronizing only at the cross-attention boundary!

---

### Fix 5: Dual Text Conditioning Pathway Extraction

Define explicit extraction in Step 1 (Qwen3-4B):
- Extract hidden states: $H_{\text{qwen}} \in \mathbb{R}^{L \times d_{\text{qwen}}}$ ($L=512, d_{\text{qwen}}=2560$).
- Sequence Conditioning: $Y_{\text{seq}} = \text{Linear}(H_{\text{qwen}}) \in \mathbb{R}^{512 \times 3072}$.
- Global Pooled Conditioning:
  $$y_{\text{pooled}} = \frac{1}{\sum m_i} \sum_{i=1}^L m_i H_{\text{qwen}}[i] \in \mathbb{R}^{2560}$$
- Timestep/Guidance Vector Modulation:
  $$c_{\text{vec}} = \text{MLP}_{\text{time}}(\text{Sinusoidal}(t)) + \text{MLP}_{\text{pool}}(y_{\text{pooled}})$$
  This vector $c_{\text{vec}}$ feeds all AdaLN-zero modulations across both Double-Stream and Single-Stream blocks.

---

## 5. Summary Scorecard

| Area | Score | Major Finding |
| :--- | :---: | :--- |
| **AdaLN-Zero Formulation** | 2 / 10 | Residual gate $\alpha$ erroneously applied to normalized input, breaking residual dynamics. |
| **RoPE Encoding** | 3 / 10 | 2D RoPE corrupts concatenated text stream; runtime trig approximations waste cycles. |
| **Flow-Matching Sampler** | 4 / 10 | Missing resolution time-shift schedule; guidance embedding path unspecified. |
| **Tensor Memory Accounting**| 3 / 10 | 16 MB L3 cache residency claim fails due to $2304 \times 3072$ patch activation expansion ($16.5\text{ MiB}$). |
| **Conditioning Pathways** | 4 / 10 | Conflates token length with dimension; omits global pooled text vector. |
| **Execution Topology** | 4 / 10 | Sub-layer CPU/GPU ping-pong causes 600+ synchronization stalls. |
| **System Lifecycle** | 9 / 10 | Sequential staging (Qwen $\to$ DiT $\to$ VAE) accurately safeguards 16 GB UMA limit. |
| **Overall** | **4.5 / 10** | **Requires major architectural revision before implementation.** |
