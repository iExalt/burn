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
| Burn fork | `41ab3d45d80a530e534b5b9ce30af59df13a165a` |
| CubeCL fork | `ac3f7686dc1777432a176a5fc71c3740cf812ddf` |
| Cubek fork | `e3f32b6369f592bc86fd6793384678c76d51de80` |

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
| cleared by direct parity | Cubek reduction | kernel reductions | forward | fp16 and f32 | Trainer isolation required deterministic `sum` and `reduce_dim` registrations, but the focused CUDA matrix clears the Cubek implementations. One-shot sum passes every advertised cube count for the observed scalar-sum shape; chained sum, vectorized output, and unit, plane, and cube routines pass TerminalO3-observed contiguous and perpendicular shapes. Continue checked trainer validation to classify higher-layer interaction. | `TerminalO3/runs/benchmarks/20260604T012835Z-phase6-autotune-reduction-contained-isolate-fused-matmul-fp16-games256/fused-matmul-fallback.yaml` | Cubek `test_terminalo3_sum_one_shot_parity`, `test_terminalo3_sum_chained_parity`, `test_terminalo3_reduce_dim_routine_parity`, and `test_terminalo3_reduce_dim_vectorized_output_parity` | not required |
| fixed and cleared by backend parity | Burn CubeCL Fusion and Cubek matmul | fused matmul | forward | fp16 and f32 | The checked accelerated selector exposed `fused_simple_vec_mat` running a general GEMM with `m > 1`, which produced correct first-row values followed by zeros. Vecmat routines now reject incompatible problems so the tuner only compares valid candidates. Checked fallback and accelerated selector workloads both pass. | `TerminalO3/runs/benchmarks/20260604T012835Z-phase6-autotune-reduction-contained-isolate-fused-matmul-fp16-games256/fused-matmul-fallback.yaml` | Burn `test_float_matmul_terminalo3_fused_accelerated_selector` and `test_float_matmul_terminalo3_fused_fallback_selector` for fp16 and f32 | Cubek `e3f32b63` |
| fixed and cleared by direct parity | Burn CubeCL and Cubek matmul | base matmul | forward | fp16 | Cyclic CMMA and MMA, TMA, output-buffer reuse, the retained large-axis transposed-RHS shape, and valid GEMV selectors pass direct CUDA parity. Vecmat selectors reject general matmul problems instead of advertising an incompatible launch. | `TerminalO3/runs/benchmarks/20260604T014355Z-phase6-autotune-contained-fp16-games256/default-cold.yaml` | Cubek `test_terminalo3_cyclic_strategy_parity`, `test_terminalo3_cyclic_large_m_axis_parity`, `test_terminalo3_tma_strategy_parity`, `test_terminalo3_gemv_selector_parity`, and `test_terminalo3_gemv_selectors_reject_general_matmul` | Cubek `e3f32b63` |
| fixed | Cubek matmul | naive fallback launch geometry | forward fallback | fp16 | Forcing the advertised base-matmul `matmul_naive` fallback first failed because its `(106624, 4, 1)` launch grid exceeded the CUDA cube-count limit. Missing handles, `Ordering is bigger than operations`, and `CallError` followed after the device worker failed. The repair spreads oversized grids and maps spread cube coordinates back to naive output elements. | `TerminalO3/runs/benchmarks/20260604T014355Z-phase6-autotune-contained-fp16-games256/isolate-base-matmul-fallback.stderr` | Cubek `test_terminalo3_naive_large_m_axis_parity` | Cubek `b1d7aeee` |

## Validation Contract

For each repaired family:

1. Add focused CUDA parity tests before changing the implementation.
2. Run the relevant Cubek CUDA crate suite.
3. Run Burn CUDA backend tests with and without fusion.
4. Clear the CubeCL autotune cache before a checked TerminalO3 cold run.
5. Run fresh-cache and cached fp16 trainer benchmarks and retain stable cache
   manifests.
