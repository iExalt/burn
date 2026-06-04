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
| Burn fork | `9d72285e6351ce4e871183629563bc09095cf69e` |
| CubeCL fork | `826c42c6f9503cdbdab126b4b69a8677582dcab5` |
| Cubek fork | `cd4e8de12dc1a517ee8474648860d8d4d5be3e95` |

Every matrix entry uses this runtime snapshot and dependency SHA set unless a
later update records an override.

## Issue Matrix

Evidence artifacts are retained in the TerminalO3 checkout under
`runs/benchmarks`. Fix commits remain `pending` until a direct parity regression
passes and the corrected candidate is available to autotune again.

| Status | Owning layer | Family | Direction | Dtype | Symptom | Evidence artifact | Regression test | Fix commit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed | Cubek convolution | implicit GEMM `SimpleAsyncTma` | forward | fp16 | Direct parity reproduced missing bias accumulation for retained `3x3` rollout shapes. The repair materializes the broadcast bias stage and uses manual MMA accumulator loads for tile geometries where `ldmatrix` addressing was incorrect. | `TerminalO3/runs/benchmarks/20260603T204209Z-phase6-all-kernel-autotune-default-fp16-games256/autotune-after-cold/0.11.0-pre.1/device-0-0-cuda/burn_cubecl-kernel-conv-forward-tune.json.log` | Cubek `test_terminalo3_forward_simple_async_tma_mma_parity` with repeated launches | Cubek `0b1876fd` |
| fixed | Cubek convolution | implicit GEMM biased CMMA setup validation | forward | fp16 | The checked trainer exposed oversized biased CMMA launches whose staged bias allocation exceeded the CUDA shared-memory budget and failed after launch. Setup now rejects those candidates before they reach the runtime while valid compact CMMA launches remain available. | `TerminalO3/runs/benchmarks/20260604T041520Z-phase6-autotune-correctness-recovery-fp16-games256/checked-empty-cache.stderr` | Cubek `test_terminalo3_forward_large_biased_cmma_shared_memory_rejected` | Cubek `cc2953bc` |
| fixed | Burn CubeCL | direct convolution accumulator | forward | fp16 and bf16 | The checked trainer exposed a large-reduction numerical mismatch because `conv_direct` accumulated half-precision products in the output dtype while matmul-backed candidates use fp32 register accumulation. Direct convolution now promotes half accumulators to fp32 and casts once when writing the output. | `TerminalO3/runs/benchmarks/20260604T041520Z-phase6-autotune-correctness-recovery-fp16-games256/checked-repaired-tma-empty-cache.stderr` | Burn `test_direct_accumulator_dtype`; checked trainer validation tracked with the TerminalO3 artifacts | Burn `43dd7ef7` |
| fixed | CubeCL runtime | checked autotune lifecycle | all | all | The checked trainer executed and copied every candidate output to the host on every cache hit, so a valid full trainer run remained operationally non-terminating after all kernel failures were repaired. Each local tuner now compares all candidates once per autotune key; persistent-cache first hits remain checked, and clearing the tuner state reruns the comparisons. This changes the check lifecycle only and does not filter, suppress, or quarantine candidates. | `TerminalO3/runs/benchmarks/20260604T041520Z-phase6-autotune-correctness-recovery-fp16-games256/checked-direct-f32-empty-cache.stderr` | CubeCL `autotune_checks_once_per_key` | CubeCL `74b88d02` |
| fixed | CubeCL runtime and Burn CubeCL Fusion | checked autotune candidate isolation | fused matmul and reductions | all | Two long-lived checked trainer runs failed at the same next fused-matmul key while isolated parity passed because correctness-check clones shared the original GPU allocations. An earlier candidate could therefore contaminate later candidates and the reference output. CubeCL now exposes a check-only clone hook, and Burn Fusion overrides it with an ordered device-side snapshot of every input allocation. Benchmark and wasm fallback clones remain shallow; no candidate is filtered, suppressed, or quarantined. | `TerminalO3/runs/benchmarks/20260604T041520Z-phase6-autotune-correctness-recovery-fp16-games256/checked-final-empty-cache.stderr` and `checked-final-rerun-empty-cache.stderr` | CubeCL `autotune_checks_once_per_key`; Burn `test_for_check_isolates_handle_allocations`, specialized fused-matmul parity, and branched linear-x backward parity | CubeCL `826c42c6`; Burn `9d72285e`; Cubek `cd4e8de1` |
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
