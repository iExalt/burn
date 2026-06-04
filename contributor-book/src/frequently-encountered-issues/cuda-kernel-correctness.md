# CUDA Kernel Correctness

CubeCL autotune candidates must be validated with direct parity tests before a
new CUDA device is treated as supported. Timing alone is not a correctness
check. Keep broad autotune runs controlled until every candidate family used by
the workload has direct parity coverage.

## Runtime Snapshot

The entries below were recorded on the host expected to be the B100 profiling
machine. CUDA reports the device name generically:

| Field | Value |
| --- | --- |
| Runtime GPU name | `NVIDIA Graphics Device` |
| Compute capability | `10.0` |
| Driver | `595.71.05` |
| Burn fork | `9a3ba5b95c6a769c8134c18a9c0c4a808c3d274b` |
| CubeCL fork | `ac3f7686dc1777432a176a5fc71c3740cf812ddf` |
| Cubek fork | `ab040c7c2e54372217f1bdd1030db5afd59414b1` |

Every matrix entry uses this runtime snapshot and dependency SHA set unless a
later update records an override.

## Issue Matrix

Evidence artifacts are retained in the TerminalO3 checkout under
`runs/benchmarks`. Fix commits remain `pending` until a direct parity regression
passes and the corrected candidate is available to autotune again.

| Status | Owning layer | Family | Direction | Dtype | Symptom | Evidence artifact | Regression test | Fix commit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed | Cubek convolution | implicit GEMM `SimpleAsyncTma` | forward | fp16 | Direct parity reproduced missing bias accumulation for retained `3x3` rollout shapes. The repair materializes the broadcast bias stage and uses manual MMA accumulator loads for tile geometries where `ldmatrix` addressing was incorrect. | `TerminalO3/runs/benchmarks/20260603T204209Z-phase6-all-kernel-autotune-default-fp16-games256/autotune-after-cold/0.11.0-pre.1/device-0-0-cuda/burn_cubecl-kernel-conv-forward-tune.json.log` | Cubek `test_terminalo3_forward_simple_async_tma_mma_parity` with repeated launches | Cubek `0b1876fd` |
| cleared by direct parity | Cubek convolution | implicit GEMM `SimpleAsyncTma` | backward-weight | fp16 | Retained learner CMMA and rollout MMA shapes pass repeated direct launches. | `TerminalO3/runs/benchmarks/20260603T204209Z-phase6-all-kernel-autotune-default-fp16-games256/autotune-after-cold/0.11.0-pre.1/device-0-0-cuda/burn_cubecl-kernel-conv-backward_weight-tune.json.log` | Cubek `test_terminalo3_backward_weight_simple_async_tma_cmma_parity` and `test_terminalo3_backward_weight_simple_async_tma_mma_parity` | not required |
| fixed | Cubek convolution | implicit GEMM `SimpleAsyncTma` | backward-data | fp16 | The stale launch guard hid an invalid TensorMap factory. The repaired factory uses input spatial shape decomposition, correct TMA swizzles, and swizzle-aware channel padding. | Cubek `crates/cubek-convolution/src/launch/base.rs` | Cubek `test_terminalo3_backward_data_simple_async_tma_mma_parity` with repeated launches | Cubek `ab040c7c` |
| fixed and cleared by direct parity | Cubek convolution | remaining implicit GEMM strategies | forward, backward-data, backward-weight | fp16 | Forward candidates shared the repaired bias stage and MMA accumulator-load issue. Sync and async cyclic and strided strategies now pass the direct parity matrix for all three directions. | `TerminalO3/runs/benchmarks/20260604T020332Z-phase6-autotune-deterministic-matmul-fp16-games256/default-cold.yaml` | Cubek `test_terminalo3_forward_non_tma_implicit_gemm_parity`, `test_terminalo3_backward_data_non_tma_implicit_gemm_parity`, and `test_terminalo3_backward_weight_non_tma_implicit_gemm_parity` | Cubek `0b1876fd` |
| needs direct parity | Cubek reduction | kernel reductions | forward | fp16 | Trainer isolation required deterministic `sum` and `reduce_dim` registrations. Direct parity must classify one-shot sum, chained sum, dimension reduction, vectorized output, and unit, plane, and cube routines. | `TerminalO3/runs/benchmarks/20260604T012835Z-phase6-autotune-reduction-contained-isolate-fused-matmul-fp16-games256/fused-matmul-fallback.yaml` | Cubek reduction parity matrix using TerminalO3-observed shapes | pending |
| needs direct parity | Burn CubeCL Fusion and Cubek matmul | fused matmul | forward | fp16 | Trainer isolation required deterministic fused-matmul selection. Selector and owned matmul candidates need direct classification. | `TerminalO3/runs/benchmarks/20260604T012835Z-phase6-autotune-reduction-contained-isolate-fused-matmul-fp16-games256/fused-matmul-fallback.yaml` | Burn fused-matmul selector regressions and Cubek matmul parity | pending |
| needs direct parity | Burn CubeCL and Cubek matmul | base matmul | forward | fp16 | Trainer isolation required deterministic base-matmul `Strategy::Auto`. Cyclic CMMA, MMA, TMA, and output-buffer reuse need direct parity. | `TerminalO3/runs/benchmarks/20260604T014355Z-phase6-autotune-contained-fp16-games256/default-cold.yaml` | Cubek matmul parity matrix using TerminalO3-observed shapes | pending |
| confirmed failure | Burn Fusion | fallback operation ordering | forward fallback | fp16 | Forcing the advertised fused-matmul fallback panics the DSU worker with `Ordering is bigger than operations`, followed by `CallError`. | `TerminalO3/runs/benchmarks/20260604T012835Z-phase6-autotune-reduction-contained-isolate-fused-matmul-fp16-games256/fused-matmul-fallback.yaml` | Burn Fusion regression that forces the fused-matmul fallback workload | pending |

## Validation Contract

For each repaired family:

1. Add focused CUDA parity tests before changing the implementation.
2. Run the relevant Cubek CUDA crate suite.
3. Run Burn CUDA backend tests with and without fusion.
4. Clear the CubeCL autotune cache before a checked TerminalO3 cold run.
5. Run fresh-cache and cached fp16 trainer benchmarks and retain stable cache
   manifests.
