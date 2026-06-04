#![allow(missing_docs)]

use super::SumAutotuneKey;
use crate::{CubeAutotuneKey, CubeRuntime, CubeTuneId, tensor::CubeTensor};
use burn_backend::cubecl::dtype_to_elem_type;
use cubecl::{
    client::ComputeClient,
    tune::{LocalTuner, Tunable, TunableSet, local_tuner},
};
use cubek::reduce::{
    ReduceDtypes, ReduceStrategy,
    components::instructions::ReduceOperationConfig,
    launch::{RoutineStrategy, VectorizationStrategy, tune_key::ReduceAutotuneKey},
    routines::{BlueprintStrategy, unit::UnitStrategy},
};

/// Executes autotune on reduce operations.
pub fn autotune_reduce<R: CubeRuntime>(
    client: &ComputeClient<R>,
    input: CubeTensor<R>,
    output: CubeTensor<R>,
    axis: usize,
    config: ReduceOperationConfig,
    dtypes: ReduceDtypes,
) {
    use reduce_ops::*;

    static TUNER: LocalTuner<ReduceAutotuneKey, CubeTuneId> = local_tuner!("reduce-dim");

    let tunables = TUNER.init(|| {
        TunableSet::new(create_key::<R>, reduce_input_gen::<R>).with(Tunable::new(
            "unit",
            |(input, output, axis, config, dtypes): (
                CubeTensor<R>,
                CubeTensor<R>,
                usize,
                ReduceOperationConfig,
                ReduceDtypes,
            )| {
                let strategy = ReduceStrategy {
                    routine: RoutineStrategy::Unit(BlueprintStrategy::Inferred(UnitStrategy)),
                    vectorization: VectorizationStrategy {
                        parallel_output_vectorization: false,
                    },
                };
                cubek::reduce::reduce::<R>(
                    &output.client,
                    input.binding(),
                    output.clone().binding(),
                    axis,
                    strategy,
                    config,
                    dtypes,
                )
                .map_err(|e| format!("{e}"))
            },
        ))
    });

    TUNER.execute(
        &CubeTuneId::new(&input.client, &input.device),
        client,
        tunables,
        (input, output, axis, config, dtypes),
    );
}

pub(crate) fn create_key<Run: CubeRuntime>(
    (input, output, axis, _config, dtypes): &(
        CubeTensor<Run>,
        CubeTensor<Run>,
        usize,
        ReduceOperationConfig,
        ReduceDtypes,
    ),
) -> ReduceAutotuneKey {
    let elem_input = dtype_to_elem_type(input.dtype);
    let elem_output = dtype_to_elem_type(output.dtype);
    let elem_acc = dtypes.accumulation.elem_type();

    ReduceAutotuneKey::generate(
        elem_input,
        elem_output,
        elem_acc,
        input.meta.shape(),
        input.meta.strides()[*axis] == 1,
        *axis,
    )
}

mod reduce_ops {
    #![allow(missing_docs)]

    use cubek::reduce::ReduceDtypes;

    use super::*;

    pub(crate) fn reduce_input_gen<Run: CubeRuntime>(
        _key: &ReduceAutotuneKey,
        (input, output, dim, config, dtypes): &(
            CubeTensor<Run>,
            CubeTensor<Run>,
            usize,
            ReduceOperationConfig,
            ReduceDtypes,
        ),
    ) -> (
        CubeTensor<Run>,
        CubeTensor<Run>,
        usize,
        ReduceOperationConfig,
        ReduceDtypes,
    ) {
        (input.clone(), output.copy(), *dim, *config, *dtypes)
    }
}

/// Executes autotune on reduce operations.
#[cfg(feature = "autotune")]
pub fn autotune_sum<R: CubeRuntime>(
    client: &ComputeClient<R>,
    input: CubeTensor<R>,
) -> CubeTensor<R> {
    use sum_ops::*;

    static TUNER: LocalTuner<CubeAutotuneKey, CubeTuneId> = local_tuner!("autotune-sum");

    let tunables = TUNER.init(|| {
        TunableSet::new(create_key_sum::<R>, sum_input_gen::<R>)
            .with(Tunable::new("sum_one_shot", sum_one_shot::<R, 4>))
    });

    TUNER.execute(
        &CubeTuneId::new(&input.client, &input.device),
        client,
        tunables,
        input,
    )
}

pub(crate) fn create_key_sum<Run: CubeRuntime>(input: &CubeTensor<Run>) -> CubeAutotuneKey {
    CubeAutotuneKey::Sum(SumAutotuneKey::generate(input))
}

impl SumAutotuneKey {
    #[allow(unused)]
    pub(crate) fn generate<Run: CubeRuntime>(input: &CubeTensor<Run>) -> Self {
        let dtype = input.dtype;
        let length = input.meta.num_elements();
        Self::new(dtype, length)
    }
}
mod sum_ops {
    #![allow(missing_docs)]
    use crate::ops::numeric::zeros_client;

    use super::*;

    pub(crate) fn sum_input_gen<Run: CubeRuntime>(
        _key: &CubeAutotuneKey,
        input: &CubeTensor<Run>,
    ) -> CubeTensor<Run> {
        input.clone()
    }

    pub(crate) fn sum_one_shot<Run: CubeRuntime, const C: u32>(
        input: CubeTensor<Run>,
    ) -> Result<CubeTensor<Run>, String> {
        let client = input.client.clone();
        let device = input.device.clone();
        let output = zeros_client(client.clone(), device, [1].into(), input.dtype);
        let dtype = input.dtype;

        cubek::reduce::shared_sum::<Run>(
            &output.client,
            input.binding(),
            output.clone().binding(),
            C,
            dtype_to_elem_type(dtype),
        )
        .map_err(|e| e.to_string())
        .map(|_| output)
    }
}
