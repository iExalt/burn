use crate::{
    CubeRuntime, CubeTuneId,
    kernel::matmul::{launch_matmul, utils::init_matmul_output},
    tensor::CubeTensor,
};
use burn_backend::DType;
use burn_backend::cubecl::dtype_to_storage_type;
use cubecl::tune::{LocalTuner, Tunable, TunableSet, local_tuner};
use cubek::matmul::strategy::{MatmulAutotuneKey, Strategy};

fn matmul_input_gen<R: CubeRuntime>(
    _key: &MatmulAutotuneKey,
    (lhs, rhs, out): &(CubeTensor<R>, CubeTensor<R>, CubeTensor<R>),
) -> (CubeTensor<R>, CubeTensor<R>, CubeTensor<R>) {
    (lhs.clone(), rhs.clone(), out.copy())
}

/// Executes autotune on matmul operations
pub fn matmul_autotune<R: CubeRuntime>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: Option<CubeTensor<R>>,
    out_dtype: DType,
) -> CubeTensor<R> {
    let output = out.unwrap_or_else(|| init_matmul_output(&lhs, &rhs, out_dtype));

    let client = lhs.client.clone();

    static TUNER: LocalTuner<MatmulAutotuneKey, CubeTuneId> = local_tuner!();

    let tunables = TUNER.init(|| {
        TunableSet::new(create_key::<R>, matmul_input_gen::<R>).with(Tunable::new(
            "matmul_auto",
            |(lhs, rhs, out)| {
                launch_matmul::<R>(&Strategy::Auto, lhs, rhs, out)
                    .map_err(|err| std::format!("{err:?}"))
            },
        ))
    });

    TUNER.execute(
        &CubeTuneId::new(&lhs.client, &lhs.device),
        &client,
        tunables,
        (lhs, rhs, output.clone()),
    );

    output
}

fn create_key<R: CubeRuntime>(
    (lhs, rhs, out): &(CubeTensor<R>, CubeTensor<R>, CubeTensor<R>),
) -> MatmulAutotuneKey {
    MatmulAutotuneKey::generate(
        &lhs.client,
        lhs.meta.shape(),
        rhs.meta.shape(),
        lhs.meta.strides(),
        rhs.meta.strides(),
        dtype_to_storage_type(lhs.dtype),
        dtype_to_storage_type(rhs.dtype),
        dtype_to_storage_type(out.dtype),
        lhs.try_scheme(),
        rhs.try_scheme(),
    )
}
