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
| Cubek fork | `1072526b66337ac123a875b426bdcd3474929400` |

Every matrix entry uses this runtime snapshot and dependency SHA set unless a
later update records an override.

## Issue Matrix

Evidence artifacts are retained in the TerminalO3 checkout under
`runs/benchmarks`. Fix commits remain `pending` until a direct parity regression
passes and the corrected candidate is available to autotune again.

| Status | Owning layer | Family | Direction | Dtype | Symptom | Evidence artifact | Regression test | Fix commit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| confirmed failure | Cubek convolution | implicit GEMM `SimpleAsyncTma` | forward | fp16 | A checked trainer run reports a `conv_autotune` numerical mismatch when the retained cache selects TMA forward kernels. The bad cache selects `simple_tma_mma` for TerminalO3 `3x3` shapes including `256 -> 64` channels. | `TerminalO3/runs/benchmarks/20260603T204209Z-phase6-all-kernel-autotune-default-fp16-games256/autotune-after-cold/0.11.0-pre.1/device-0-0-cuda/burn_cubecl-kernel-conv-forward-tune.json.log` | Cubek forward parity for observed TMA CMMA and MMA shapes, including repeated launches | pending |
| needs direct parity | Cubek convolution | implicit GEMM `SimpleAsyncTma` | backward-weight | fp16 | The bad cache selects `simple_tma_cmma` and `simple_tma_mma` for learner gradient shapes. Candidate correctness has not been isolated from the degenerate trainer result. | `TerminalO3/runs/benchmarks/20260603T204209Z-phase6-all-kernel-autotune-default-fp16-games256/autotune-after-cold/0.11.0-pre.1/device-0-0-cuda/burn_cubecl-kernel-conv-backward_weight-tune.json.log` | Cubek backward-weight parity for observed TMA CMMA and MMA shapes, including repeated launches | pending |
| confirmed unavailable | Cubek convolution | implicit GEMM `SimpleAsyncTma` | backward-data | fp16 | Cubek explicitly rejects this strategy because its current TMA tiling path is not implemented correctly for data gradients. | Cubek `crates/cubek-convolution/src/launch/base.rs` | Cubek backward-data parity for observed TMA MMA shapes | pending |
| needs direct parity | Cubek convolution | remaining implicit GEMM strategies | forward, backward-data, backward-weight | fp16 | Removing the remaining implicit-GEMM candidates was required before the trainer stayed healthy. Individual candidates still need direct classification. | `TerminalO3/runs/benchmarks/20260604T020332Z-phase6-autotune-deterministic-matmul-fp16-games256/default-cold.yaml` | Cubek convolution parity matrix for all registered non-TMA implicit-GEMM strategies | pending |
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
