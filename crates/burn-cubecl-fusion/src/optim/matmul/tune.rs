use super::optimization::MatmulOptimizationTuneArg;
use crate::{
    CubeFusionHandle,
    engine::trace::TuneOutput,
    tune::{FusionInputGen, TuneInput},
};
use burn_backend::cubecl::dtype_to_storage_type;
use burn_fusion::stream::Context;
use cubecl::{
    AutotuneKey, CubeTuneId, Runtime,
    tune::{LocalTuner, Tunable, TunableSet, local_tuner},
};
use cubek::matmul::strategy::MatmulAutotuneKey;
use serde::{Deserialize, Serialize};

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
pub struct FusedMatmulAutotuneKey {
    matmul_key: MatmulAutotuneKey,
    #[autotune(anchor)]
    num_out_buffers: usize,
    #[autotune(anchor)]
    num_ops: usize,
}

/// Executes autotune on matmul operations
pub fn fused_matmul_autotune<R: Runtime>(
    optimization: MatmulOptimizationTuneArg<R>,
    context: &mut Context<CubeFusionHandle<R>>,
) {
    static TUNER: LocalTuner<FusedMatmulAutotuneKey, CubeTuneId> = local_tuner!();

    let tunables = TUNER.init(|| {
        TunableSet::new(create_key::<R>, FusionInputGen)
            .with(Tunable::new("fused_matmul_fallback", tune_fallback::<R>))
    });

    TUNER.execute(
        &CubeTuneId::new(&optimization.info.client, &optimization.info.device),
        &optimization.info.client.clone(),
        tunables,
        TuneInput::new(context, optimization),
    );
}

pub(crate) fn create_key<R: Runtime>(
    input: &TuneInput<R, MatmulOptimizationTuneArg<R>>,
) -> FusedMatmulAutotuneKey {
    let opt = input.optimization();
    assert!(input.is_original(), "Not supported when generating key");
    let tensors = input.tensors();
    let handles = input.handles();

    let lhs = tensors.get(&opt.info.matmul.op.lhs.id).unwrap();
    let rhs = tensors.get(&opt.info.matmul.op.rhs.id).unwrap();
    let out = tensors.get(&opt.info.matmul.op.out.id).unwrap();

    let lhs_strides = handles
        .get_handle_ref(&lhs.id)
        .expect("lhs handle")
        .strides
        .clone();
    let rhs_strides = handles
        .get_handle_ref(&rhs.id)
        .expect("rhs handle")
        .strides
        .clone();

    let key = MatmulAutotuneKey::generate(
        &opt.info.client,
        &lhs.shape,
        &rhs.shape,
        &lhs_strides,
        &rhs_strides,
        dtype_to_storage_type(lhs.dtype),
        dtype_to_storage_type(rhs.dtype),
        dtype_to_storage_type(out.dtype),
        opt.info.matmul.lhs.scheme(),
        opt.info.matmul.rhs.scheme(),
    );
    FusedMatmulAutotuneKey::new(key, opt.info.num_output_buffers(), opt.info.num_ops_fused())
}

fn tune_fallback<R: Runtime>(
    input: TuneInput<R, MatmulOptimizationTuneArg<R>>,
) -> Result<TuneOutput<R>, String> {
    Ok(input.execute(|ctx, opt| opt.execute_fallback(ctx)))
}
