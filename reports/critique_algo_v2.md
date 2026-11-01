# DiT Algorithm & Diffusion Review: FLUX.2 Klein 4B Megakernel Plan (Version 2.0)

**Critic:** Critic 2 — DiT Algorithm & Diffusion Specialist  
**Evaluation Round:** Round 2 (Post-Revision Verification)  
**Target Plan:** `c:\Users\abhik\Documents\antigravity\silly-pascal\scratch\flux2_dit_megakernel_plan_v2.md`  
**System Target:** AMD Ryzen 7 7730U APU (Zen 3 Barcelo-R, Vega 8 `gfx90c`, 16 GB UMA DDR4-3200)  
**Workload Target:** `flux-2-klein-4b-Q3_K_M.gguf` (57 blocks: 19 double-stream + 38 single-stream), `qwen3-4b-Q4_K_M.gguf`, 16-channel VAE, 4-step flow-matching Euler sampler ($768\times 768$).

---

## 1. Executive Evaluation & Score

### **Overall Score: 9.3 / 10** (Up from 4.5 / 10 in Round 1)

**Verdict:** *Fully Resolved, Mathematically Sound, and Production-Ready.*  
The Version 2.0 Megakernel Plan has systematically addressed every critical mathematical and algorithmic failure mode identified in Round 1. The confusing and hazardous in-flight scaling has been replaced with a rigorous AdaLN-zero decoupling that honors identity initialization. The runtime polynomial trigonometric approximation has been replaced with a static 704 KiB 3D RoPE look-up table with exact coordinate partitioning. The empirical resolution-shifted flow-matching time schedule $\mu(N)$ is properly established for 4-step distilled Euler integration, and the dual text conditioning pathways ($512 \times 3072$ sequence tokens + $y_{\text{pooled}}$ vector) are correctly routed.

Furthermore, moving 100% of the 57 DiT blocks to Vega 8 with FlashAttention-2 in 48 KB LDS resolves both the impossible 16 MB L3 cache residency claim and the 600+ synchronization barrier ping-pong trap.

---

## 2. Verification of Round 1 Algorithmic Critiques

| Review Item | Round 1 Critical Flaw | Round 2 Resolution Status | Assessment & Verification |
| :--- | :--- | :---: | :--- |
| **1. AdaLN-Zero Formulation** | Applied residual gate $\alpha$ directly to normalized input tokens alongside $(\gamma, \beta)$, breaking identity initialization and causing activation divergence. | **VERIFIED & RESOLVED** | Strict mathematical decoupling: pre-norm affine scaling $\hat{x}(1+\gamma)+\beta$ is isolated within the pre-layer sweep, while $\alpha$ strictly gates the sublayer output prior to residual addition ($x_{l+1} = x_l + \alpha \odot \text{SubLayer}(\dots)$). |
| **2. 3D RoPE Positional Encoding** | Dynamic polynomial `sin`/`cos` computed across concatenated tokens without coordinate masking, corrupting text features with 2D spatial frequencies. | **VERIFIED & RESOLVED** | Precomputes a static **704 KiB 3D RoPE LUT** in UMA. Text tokens are assigned $(i, 0, 0)$ and image patches are assigned $(0, h, w)$, correctly partitioning the 128-dim head into dedicated frequency subspaces with zero runtime trig overhead. |
| **3. Flow-Matching Euler Schedule** | Naive linear timestep schedule omitted resolution-dependent trajectory bending; guidance conditioning was omitted. | **VERIFIED & RESOLVED** | Incorporates empirical time shift $t' = \frac{\mu t}{1 + (\mu - 1)t}$ with $\mu = \mu(N)$, aligning 4-step discrete Euler knots with the distilled flow-matching vector field. |
| **4. Dual Text Pathways** | Conflated sequence length (512) with feature dimension; omitted global pooled text vector $y_{\text{pooled}}$ required for AdaLN conditioning. | **VERIFIED & RESOLVED** | Formulates dual output from Qwen3-4B: (1) $512 \times 3072$ sequence tokens for joint cross-attention, and (2) $y_{\text{pooled}}$ global context vector routed into the timestep modulation MLP. |
| **5. Latent Geometry & Patching** | Claimed $16\times 96\times 96$ latent (miscalculated as 147 KB) remained L3-resident, ignoring $2\times 2$ patch expansion ($2304 \times 3072 = 13.5\text{ MiB}$). | **VERIFIED & RESOLVED** | Correctly models $2\times 2$ patch packing to 2,304 image tokens + 512 text tokens ($S=2816$). Replaced L3 CPU cache residency with **FlashAttention-2 tiled in 32 KB LDS per CU** on Vega 8 ($O(S)$ memory). |
| **6. Execution Topology** | Sub-layer CPU/GPU ping-pong across 57 blocks $\times$ 4 steps incurred 600+ synchronization fences and WDDM command queue bubbles. | **VERIFIED & RESOLVED** | Unified **100% of DiT execution in an in-order OpenCL queue on Vega 8**. Zero CPU-GPU sync bubbles during the 4 denoising steps. |

---

## 3. In-Depth Algorithmic Analysis of Plan v2.0

### A. Mathematical Correctness of AdaLN-Zero Decoupling
In Section 1 and Section 2.C.1, the modulation is now partitioned into two distinct operations:
1. **Pre-Sublayer Input Normalization & Affine Scaling:**
   $$\hat{x} = \text{RMSNorm}(x), \quad x_{\text{in}} = \hat{x} \odot (1 + \gamma) + \beta$$
2. **Post-Sublayer Residual Gating:**
   $$x_{\text{out}} = x + \alpha \odot \text{SubLayer}(x_{\text{in}})$$
This preserves the critical zero-initialization boundary condition: when $\alpha \to 0$, $x_{\text{out}} = x$, ensuring numerical stability through the 57 layers of FLUX.2 Klein 4B.

### B. 3D RoPE Table Sizing & Multi-Axis Partitioning
The 704 KiB static RoPE LUT is mathematically exact:
- **Total Sequence Length:** $S = 512 \text{ (text)} + (48 \times 48) \text{ (image patches)} = 2,816$ tokens.
- **Head Dimension:** $d_k = 128 \implies 64$ complex rotary pairs.
- **Memory Footprint:** 
  $$2,816 \text{ tokens} \times 64 \text{ pairs} \times 2 \ (\cos, \sin) \times 2\text{ bytes (FP16)} = 720,896\text{ bytes} \approx 704.0\text{ KiB}$$
- **Coordinate Assignment:**
  - For $i \in [0, 511]$ (Text tokens): $\text{coord}(i) = (i, 0, 0)$. Rotary frequencies are active only on the text frequency channels ($d_t = 32$), while spatial channels receive zero rotation ($\cos = 1, \sin = 0$).
  - For $j = h \cdot 48 + w \in [0, 2303]$ (Image tokens): $\text{coord}(j) = (0, h, w)$. Text frequency channels receive zero rotation, while vertical ($d_h = 48$) and horizontal ($d_w = 48$) channels rotate according to 2D spatial grid coordinates.
This prevents cross-modal positional interference and enables bidirectional joint attention over the unified sequence.

### C. Resolution-Shifted Flow-Matching Sampler
The integration trajectory follows the Black Forest Labs shifted ODE:
$$t_{\text{shifted}} = \frac{\mu(N) \cdot t}{1 + (\mu(N) - 1) \cdot t}$$
For $N = 2,304$ image tokens, $\mu(N) \approx 1.05$.  
Discrete integration knots for the 4-step Euler sampler:
$$t_k \in \{1.0, 0.75, 0.50, 0.25, 0.0\} \xrightarrow{\mu(N)} \{1.0, 0.759, 0.512, 0.259, 0.0\}$$
The Euler step:
$$x_{k+1} = x_k + (t'_{k+1} - t'_k) \cdot v_\theta(x_k, t'_k)$$
This non-linear warping accurately tracks the velocity vector field $v_\theta$, preventing the contrast collapse and high-frequency noise that plague linear 4-step schedules.

### D. FlashAttention-2 in 48 KB Vega 8 LDS
Moving from unrolled attention to FlashAttention-2 inside the Compute Unit LDS completely eliminates the $380\text{ MB}$ intermediate attention matrix:
- Tile dimensions: $B_r = 64, B_c = 64, d_k = 128$.
- LDS allocation per CU:
  $$Q_{\text{tile}} + K_{\text{tile}} = 64 \times 128 \times 2\text{ B} \times 2 = 32\text{ KB} \le 48\text{ KB}$$
- Online softmax accumulation maintains running scalar maximum $m$ and normalization sum $l$ in 32-bit VGPRs, guaranteeing numeric stability without underflow.

---

## 4. Minor Implementation Guidelines for Version 2.0 Execution

To ensure seamless implementation during kernel coding, the engineering team should observe three minor operational nuances:

1. **Double-Stream vs Single-Stream Modulation Dimensions:**
   - Double-stream blocks (Blocks 0–18) have separate MLPs for text ($12\times d$) and image ($12\times d$), yielding 6 modulation parameters each ($\gamma_1, \beta_1, \alpha_1, \gamma_2, \beta_2, \alpha_2$).
   - Single-stream blocks (Blocks 19–56) operate on the unified $S=2816$ sequence with unified modulation parameters ($6\times d$). Ensure the modulation kernel offsets indices appropriately when transitioning from Block 18 to Block 19.
2. **Euler Integration Direction Sign:**
   - In FLUX conventions, $t=1.0$ is Gaussian noise and $t=0.0$ is clean data. Thus, $\Delta t = (t'_{k+1} - t'_k) < 0$. Verify that the accumulator computes $x_k + \Delta t \cdot v_\theta$ (or $x_k - |\Delta t| \cdot v_\theta$) to avoid adding noise instead of subtracting it.
3. **Double-Stream Attention Masking:**
   - In Double-Stream blocks, queries from image tokens attend to both text and image keys ($[K_{\text{txt}}; K_{\text{img}}]$), and queries from text tokens attend to both ($[K_{\text{txt}}; K_{\text{img}}]$). Since DiT attention is fully bidirectional, no causal mask is applied. Ensure the FlashAttention-2 kernel loops over all $S=2816$ keys for all queries.

---

## 5. Final Algorithmic Scorecard

| Evaluated Dimension | Plan v1.0 | Plan v2.0 | Assessment |
| :--- | :---: | :---: | :--- |
| **AdaLN-Zero Decoupling** | 2 / 10 | **10 / 10** | Mathematically rigorous; separates pre-norm scale/shift from residual gate. |
| **RoPE Coordinate Encoding** | 3 / 10 | **9.5 / 10** | Exact 704 KiB static LUT with 3D coordinate partitioning ($(i,0,0)$ and $(0,h,w)$). |
| **Flow-Matching Sampler** | 4 / 10 | **9.5 / 10** | Resolution-dependent non-linear time shift $\mu(N)$ fully integrated into 4-step Euler. |
| **Text Conditioning Pathways** | 4 / 10 | **9.5 / 10** | Dual extraction ($512\times 3072$ sequence tokens + $y_{\text{pooled}}$ AdaLN vector). |
| **Attention & Memory Complexity** | 3 / 10 | **9.5 / 10** | FlashAttention-2 in 32 KB LDS eliminates 380 MB buffer and respects CU limits. |
| **Execution Topology** | 4 / 10 | **9.0 / 10** | 100% Vega 8 in-order execution completely eliminates CPU-GPU sync stalls. |
| **Overall Algorithmic Score** | **4.5 / 10** | **9.3 / 10** | **Fully Approved for Implementation.** |
