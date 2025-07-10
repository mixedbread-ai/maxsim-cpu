# MaxSim CPU Performance: Reranking 1000 Documents

**Date**: 2025-07-10 14:32:56

**CPU**: Apple M4 Max

**Task**: Reranking 1000 documents using MaxSim

**Query Dimensions**: 32 x 128

**Iterations per Test**: 50

## Performance Summary

### Documents Processed per Second

| Method | 512 tokens | 1024 tokens | 2048 tokens | Variable (128-1536) |
|--------|-----------|-----------|-----------|-----------|
| JAX |   229,382 |    99,670 |    38,956 |    13,006 |
| JAX (vmap) |   227,550 |    99,051 |    38,710 |    13,137 |
| NumPy |   100,015 |    42,639 |    21,377 |    10,492 |
| PyTorch (baseline) |   164,256 |    59,888 |    30,238 |    12,089 |
| PyTorch (matmul) |   173,633 |    60,685 |    30,432 |    11,887 |
| Rust (padded) | N/A | N/A | N/A |    15,450 |
| maxsim-cpu (Rust) |   323,549 |   169,730 |    87,152 | N/A |
| **Rust Variable API** | N/A | N/A | N/A | ** 60,723** |

### Speedup vs PyTorch Baseline

| Method | 512 tokens | 1024 tokens | 2048 tokens | Variable (128-1536) |
|--------|-----------|-----------|-----------|-----------|
| JAX | **1.40x** | **1.66x** | **1.29x** | **1.08x** |
| JAX (vmap) | **1.39x** | **1.65x** | **1.28x** | **1.09x** |
| NumPy | 0.61x | 0.71x | 0.71x | 0.87x |
| PyTorch (baseline) | **1.00x** | **1.00x** | **1.00x** | **1.00x** |
| PyTorch (matmul) | **1.06x** | **1.01x** | **1.01x** | 0.98x |
| Rust (padded) | N/A | N/A | N/A | **1.28x** |
| Rust Variable API | N/A | N/A | N/A | **5.0x** |
| maxsim-cpu (Rust) | **1.97x** | **2.83x** | **2.88x** | N/A |

## Key Findings

- **maxsim-cpu (Rust)** achieves **2.6x** average speedup over PyTorch baseline on fixed lengths
  - Speedup range: 2.0x to 2.9x

- **Variable Length Performance** (Real-world scenario):
  - Rust Variable API achieves **5.0x** speedup over PyTorch
  - PyTorch requires padding to 1536 tokens (average doc length: 852)
  - This results in ~80% wasted computation for PyTorch
  - Rust processes actual document lengths without padding overhead

## System Configuration

- Python: 3.13.5
- PyTorch: 2.7.1
- NumPy: 2.3.1
- Platform: macOS-15.5-arm64-arm-64bit-Mach-O
