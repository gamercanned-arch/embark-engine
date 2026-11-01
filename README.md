# EMBARK Engine
### Heterogeneous APU Inference Engine for AMD Ryzen 7 7730U (Zen 3 + Radeon Vega 8)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%2011%20%7C%20Linux-lightgrey.svg)]()
[![Target Architecture](https://img.shields.io/badge/Arch-AMD%20Zen%203%20%2B%20Vega%208%20(gfx90c)-informational.svg)]()

EMBARK is an independent, bare-metal C++17 heterogeneous inference engine designed for AMD Accelerated Processing Units (APUs) utilizing unified memory architecture (UMA). 

By co-designing execution paths across the **AMD Zen 3 CCX** (8 cores, 16 threads, 16 MB unified L3 cache) and the **Radeon Vega 8 iGPU** (8 CUs @ 2000 MHz, GCN 5.0 / `gfx90c`, Wave64), EMBARK maximizes hardware utilization on consumer APUs across DDR4-3200 memory streaming and Rapid Packed Math (RPM) compute pipelines.

---

## Architecture Overview

```
                          ┌───────────────────────────────────────┐
                          │   EMBARK APU Heterogeneous Engine     │
                          └───────────────────┬───────────────────┘
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    ▼                                                   ▼
       ┌────────────────────────┐                          ┌────────────────────────┐
       │   Radeon Vega 8 iGPU   │                          │    Zen 3 8-Core CPU    │
       │    (Prompt Prefill)    │                          │     (Token Decode)     │
       ├────────────────────────┤                          ├────────────────────────┤
       │ • OpenCL Wave64 RPM    │                          │ • AVX2 + FMA Interleave│
       │ • FP16 LDS Ping-Pong   │                          │ • 8-Core Physical Pin  │
       │ • 2-3ms WDDM Slicing   │                          │ • Zero-Barrier MoE Rtr │
       │ • 2.465 TFLOPS Sust.   │                          │ • 23.75 GB/s Streaming │
       └────────────────────────┘                          └────────────────────────┘
                    │                                                   │
                    └─────────────────────────┬─────────────────────────┘
                                              ▼
                               ┌────────────────────────┐
                               │  16 GB DDR4-3200 UMA   │
                               │ Zero-Copy Shared Pool  │
                               └────────────────────────┘
```

### 1. Dual-Engine Heterogeneous Scheduling
* **Prompt Prefill on iGPU**: GEMM-intensive prompt evaluation is scheduled on the Vega 8 iGPU using OpenCL Rapid Packed Math (RPM) kernels operating on `half2` vector registers, sustaining **2.465 TFLOPS** (60.2% of theoretical hardware saturation).
* **Token Decode on Zen 3 CPU**: Memory bandwidth-bound autoregressive decoding is processed entirely on the 8 physical Zen 3 cores using vectorized AVX2 dot-product microkernels.
* **Zero-Copy Host Pointer Buffers**: Eliminates PCIe and system bus copy overhead by allocating weights directly in unified DDR4-3200 memory shared between CPU and iGPU.

### 2. Zen 3 Core Pinning (`cpumask=0x5555`)
* Pins execution threads to the 8 physical execution cores (`0x5555` = cores `0, 2, 4, 6, 8, 10, 12, 14`).
* Prevents SMT sibling contention and cache line invalidation across the unified 16 MB CCX L3 slice, minimizing L3 access latency.

### 3. Dual-Issue FMA Pipelining (Ports 0 and 1)
* Decode microkernels implement 4-way accumulator unrolling (`_mm256_fmadd_ps` across `ymm0`–`ymm7`) to utilize Zen 3's dual 256-bit FMA execution units located on execution ports 0 and 1.
* Vectorized `quantize_row_q8_K` and AVX2 interleaved dot products achieve **23.75 GB/s** memory streaming throughput (76% of DDR4-3200 dual-channel theoretical limit).

### 4. Zero-Barrier MoE Routing and Memory Scatter Mitigation
* Eliminates up to 456 thread synchronization barriers per token present in stock frameworks during `MUL_MAT_ID` expert dispatch via a direct 1-thread-per-expert topology.
* Contiguous expert memory staging (`[Gate | Up | Down]`) mitigates the -21.05% bandwidth penalty caused by random DDR4 pointer hopping.

### 5. Cooperative WDDM Compute Slicing (`--perf-mode high | xhigh`)
* Prevents desktop stutter and Windows Desktop Window Manager (DWM) starvation by slicing long GPU prefill dispatches into 2–3ms cooperative compute tiles:
  * `--perf-mode high` (Default): Cooperative WDDM timeslicing; desktop UI and audio pipelines remain responsive.
  * `--perf-mode xhigh`: Maximum sustained prefill dispatch; saturates all 8 Compute Units continuously.

### 6. Native AMD OpenCL Backend
* Resolves an upstream issue where AMD APUs were incorrectly assigned Qualcomm Adreno GPU configurations, leading to incompatible texture swizzling operations.
* Enforces native `GPU_FAMILY::AMD` with Wave64 subgroup execution, LDS double-buffering, and host-memory direct addressing.

---

## Benchmark Results (AMD Ryzen 7 7730U)

All measurements conducted on standalone EMBARK engine binaries (`build/*_apu_engine.exe`) under Windows 11 with 16 GB DDR4-3200 UMA.

> [!NOTE]
> **Measurement Methodology**:
> * **Token Decode (`tg128`)**: 100% end-to-end multi-layer execution on real weights. Zen 3 physical cores actively stream expert slices from DDR4 RAM and evaluate all router projections and active expert paths sequentially.
> * **Prompt Prefill (`pp512`)**: Evaluates sustained Vega 8 Rapid Packed Math (RPM) GPU compute on projection GEMMs and maps across the model layer forward graph. For MoE architectures, actual batched prefill throughput is governed by token dispersion across the 64 experts ($M \approx 32$ per expert, ~30 tok/s sustained), whereas idealized dense batching scales up to theoretical compute limits.

| Model Architecture | Quantization | Engine Binary | pp512 Prefill | tg128 Decode | Peak Burst | Effective Memory Bandwidth |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LFM 2.5 350M** | `Q8_0` | `lfm2_350m_apu_engine` | **481.40 tok/s** (vs 155.97) | **63.72 tok/s** (vs 40.26) | 71.34 tok/s | 22.36 GB/s |
| **Spark-X 2.6 4B** | `Q4_K_M` | `spark_apu_engine` | **41.24 tok/s** (vs 23.37) | **6.86 tok/s** (vs 5.32) | 7.44 tok/s | 22.60 GB/s |
| **LFM3-VL-2B** (Vision-Lang) | `Q4_K_M` | `lfm3_apu_engine` | **101.23 tok/s** (vs 62.58) | **20.21 tok/s** (vs 12.80) | 23.38 tok/s | 20.55 GB/s (Vision: 27.02 ms) |
| **Gemma 4 E4B** | `Q4_K_M` | `gemma4_e4b_apu_engine` | **23.91 tok/s** (vs 22.37) | **5.09 tok/s** (vs 4.69) | 5.50 tok/s | 23.50 GB/s |
| **Ornith 1.5 9B** | `Q4_K_M` | `ornith_apu_engine` | **20.33 tok/s** (vs 12.21) | **4.17 tok/s** (vs 2.91) | 4.52 tok/s | 22.42 GB/s (GDN in L2 cache) |
| **Gemma 4 12B** | `Q4_K_M` | `gemma4_apu_engine` | **15.59 tok/s** (vs 7.32) | **3.21 tok/s** (vs 1.82) | 3.63 tok/s | 21.99 GB/s |
| **Groot 19B MoE** | `IQ3_XXS` | `groot_apu_engine` | **27.50 tok/s** (Dispersion) | **27.19 tok/s** (vs 3.85) | 31.72 tok/s | L3 Resident (37.5% Hit Rate) |
| **d1-2B** (Decision/Vision) | `Q4_K_M` | `d1_apu_engine` | **99.06 tok/s** (vs 57.78) | **21.07 tok/s** (vs 11.75) | 23.85 tok/s | 21.42 GB/s (Vision: 27.99 ms) |
| **EmbeddingGemma-2** | `Q6_K_XL` | `embedding_apu_engine` | **363.56 tok/s** (Forward) | **464.60 tok/s** (E2E Multi) | 75.15 ms mmproj | 0.79 TFLOPS sustained |
| **LFM 2.5 24B MoE** | `IQ3_XXS` | `zen3_apu_engine` | **30.10 tok/s** (Dispersion) | **35.00 tok/s** (vs 3.13) | 45.47 tok/s | 16.68 GB/s Contiguous (7.81 GB Pool) |
| **LFM 2.5 24B MoE** | `Q3_K_XL` | `zen3_apu_engine` | **30.10 tok/s** (Dispersion) | **29.74 tok/s** (vs 3.13) | 35.34 tok/s | 17.58 GB/s Contiguous (9.46 GB Pool) |

---

## Microkernel Performance and Memory Analysis

GEMM and streaming bandwidth evaluation on AMD Vega 8 (`gfx90c`) and Zen 3 cores:

```
Kernel Implementation                  Throughput / Bandwidth   Implementation Details
───────────────────────────────────────────────────────────────────────────────────────
bench_master_unified_rpm.cpp           2.465 TFLOPS             Wave64 FP16 LDS Ping-Pong
bench_ultimate_rpm.cpp                 2.380 TFLOPS             Double-buffered LDS co-issue
bench_pipelined_fma.cpp                2.110 TFLOPS             Dual-issue port 0/1 FMA
bench_transposed_vector_gemm.cpp       2.040 TFLOPS             Transposed 4x4 register tile
bench_vector_lds.cpp                   1.890 TFLOPS             Vectorized LDS load/store
bench_vega8_saturation.cpp             1.624 TFLOPS             Continuous compute saturation
moe_router_benchmark.cpp               23.75 GB/s               Zero-barrier contiguous expert stream
mt_expert_stream.cpp                   23.40 GB/s               Multi-threaded expert prefetching
```

### MoE Memory Access Analysis
* **Contiguous 4-Expert Stream**: `23.75 GB/s`
* **Scattered 4-Expert Stream**: `18.75 GB/s` (-21.05% bandwidth penalty)
* **Projection-Hopping Stream**: `19.12 GB/s` (-19.50% bandwidth penalty)
* **Observation**: Sub-byte `IQ2_S` / `IQ3_S` formats utilize scalar codebook lookups (~16 scalar memory reads per 64 weights), causing execution port contention. Staging weights into contiguous layouts and utilizing vectorized SIMD paths restores full DRAM streaming bandwidth.

---

## Repository Structure

```
embark-engine/
├── CMakeLists.txt              # Build configuration (AVX2 + OpenCL)
├── LICENSE                     # MIT License
├── README.md                   # Technical documentation and benchmarks
├── include/                    # Self-contained OpenCL C/C++ SDK headers
│   └── CL/
├── src/                        # 11 Dedicated Model Engines
│   ├── zen3_engine/            # Zen 3 APU orchestrator & OpenCL testbed
│   ├── lfm2_350m_engine/       # Dual-engine orchestrator, threadpool, tiler
│   ├── groot_engine/           # Groot 19B MoE engine (SIMD, tiler, OpenCL)
│   ├── spark_engine/           # Spark-X 4B engine
│   ├── gemma4_engine/          # Gemma 4 12B engine
│   ├── gemma4_e4b_engine/      # Gemma 4 E4B engine
│   ├── ornith_engine/          # Ornith 9B engine
│   ├── lfm3_vl_engine/         # LFM3-VL 2B multimodal engine
│   ├── d1_engine/              # d1 2B decision engine
│   ├── embedding_gemma_engine/ # EmbeddingGemma 2048-dim engine
│   └── flux2_dit_engine/       # FLUX.2 Klein 4B Diffusion Transformer engine
├── benchmarks/
│   ├── cpp/                    # 17 C++ GEMM, RPM, and MoE microkernels
│   └── scripts/                # 35 Analysis and inspection scripts
├── patches/                    # Patches for upstream integrations
│   ├── embark_llama_cpp.patch  # AMD GPU_FAMILY, Wave64, cpumask=0x5555
│   ├── embark_sd_cpp.patch     # SD.cpp APU build & cfgmgr32 integration
│   ├── embark_sd_ggml.patch    # SD GGML AVX2 quantize_row_q8_K optimization
│   ├── tuned_engine.patch      # Tuned engine diffs
│   └── tuned_quants.patch      # AVX2 interleaved dot product optimizations
├── scripts/                    # Compilation batch scripts (MSVC)
├── reports/                    # Engineering and microarchitecture reports
├── metadata/                   # Model tensor and layer metadata JSONs
└── docs/                       # Architectural diagrams and plans
```

---

## Build and Installation

### Prerequisites
* **Operating System**: Windows 10/11 64-bit or Linux x86_64
* **Compiler**: Microsoft Visual Studio 2022+ (MSVC `vcvars64.bat`) or GCC 11+ / Clang 14+
* **Processor Requirements**: x86_64 with AVX2, F16C, FMA support
* **OpenCL Runtime**: Self-contained headers included; links against system `OpenCL.dll` or `OpenCL_sys.lib`

### Building via Batch Scripts (Windows)
Each engine can be compiled independently:
```cmd
cd scripts
build_lfm2_350m.bat
build_groot.bat
build_spark.bat
```

### Building via CMake
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

### Running Microkernel Benchmarks
```cmd
# Run Vega 8 Wave64 RPM compute benchmark
.\build\Release\bench_master_unified_rpm.exe

# Run MoE contiguous memory streaming benchmark
.\build\Release\moe_router_benchmark.exe
```

---

## Tokenizer Compatibility

EMBARK engines include an embedded unified tokenizer implementation:

| Model Family | Tokenizer Type | Pre-tokenizer | Vocab Size | Models |
| :--- | :---: | :---: | :---: | :--- |
| **Gemma 4** | SentencePiece / Gemma | None | 524,293 | Gemma 4 12B, Gemma 4 E4B, EmbeddingGemma-2 |
| **Qwen / Groot** | BPE / GPT-2 | Qwen 3.5 | 496,645 | Groot 19B, Ornith 9B, Qwen 3.5 4B |
| **LFM 2 / 3** | BPE / GPT-2 | LFM2 | 256,005 / 131,077 | LFM 2.5 24B, LFM3-VL 2B, LFM 2.5 350M, d1-2B |
| **Spark-X** | BPE / GPT-2 | Spark2.5 | 262,149 | Spark-X 4B |
| **Qwen 3** | BPE / GPT-2 | Qwen2 | 303,877 | Qwen 3 4B |

---

## License
Distributed under the [MIT License](LICENSE). Copyright © 2026 Abhik and EMBARK Contributors.
