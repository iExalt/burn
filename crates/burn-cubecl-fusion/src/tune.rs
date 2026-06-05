use crate::CubeFusionHandle;
use burn_fusion::stream::Context;
use burn_ir::{HandleContainer, TensorId, TensorIr};
use cubecl::Runtime;
use cubecl::tune::{InputGenerator, TuneInputs};
use hashbrown::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;

/// [`TuneInputs`] marker for [`TuneInput`]. This is the indirection that lets a
/// `TunableSet<_, FusionTuneInputs<R, O>, _>` live `'static` inside `LocalTuner::init`'s
/// cache while still accepting a borrowing `TuneInput<'a, …>` at `execute` time, via HRTB
/// over `'a`.
#[allow(clippy::type_complexity)]
pub(crate) struct FusionTuneInputs<R: Runtime, O>(PhantomData<(fn() -> R, fn() -> O)>);

impl<R: Runtime, O: Send + Sync + 'static> TuneInputs for FusionTuneInputs<R, O> {
    type At<'a> = TuneInput<'a, R, O>;

    #[cfg(feature = "autotune-checks")]
    fn clone_for_check<'a>(inputs: &Self::At<'a>) -> Self::At<'a> {
        inputs.for_check()
    }
}

/// [`InputGenerator`] for [`TuneInput`]: produces a benchmark-only [`TuneState::Fork`]
/// (i.e. with `new_handles = None`). Benchmarks discard their outputs, so the returned
/// input skips the handle-tracking machinery that a wasm-fallback fork carries.
pub(crate) struct FusionInputGen;

impl<K, R, O> InputGenerator<K, FusionTuneInputs<R, O>> for FusionInputGen
where
    K: 'static,
    R: Runtime,
    O: Send + Sync + 'static,
{
    fn generate<'a>(
        &self,
        _key: &K,
        inputs: &<FusionTuneInputs<R, O> as TuneInputs>::At<'a>,
    ) -> <FusionTuneInputs<R, O> as TuneInputs>::At<'a> {
        inputs.for_benchmark()
    }
}

/// Shared staging area for handles produced by a wasm try-all fork.
///
/// [`Fork`](TuneState::Fork)'s `Drop` dumps all of the fork's handles here (replacing any
/// previous contents), and [`Original`](TuneState::Original)'s `Drop` drains this and
/// filters each entry against the real context's current handles — anything the real
/// context doesn't already have is a genuinely new output and gets registered.
///
/// The pipeline is strictly serial, so
/// the lock is always uncontended; `spin::Mutex` is there purely to give
/// `Arc<HandleCollector<R>>` `Send + Sync`.
pub(crate) struct HandleCollector<R: Runtime>(spin::Mutex<HashMap<TensorId, CubeFusionHandle<R>>>);

impl<R: Runtime> HandleCollector<R> {
    fn new() -> Self {
        Self(spin::Mutex::new(HashMap::new()))
    }

    fn capture(&self, handles: &HandleContainer<CubeFusionHandle<R>>) {
        let mut bag = self.0.lock();
        bag.clear();
        for id in handles.handle_ids() {
            if let Some(h) = handles.get_handle_ref(id) {
                bag.insert(*id, h.clone());
            }
        }
    }

    fn take(&self) -> HashMap<TensorId, CubeFusionHandle<R>> {
        core::mem::take(&mut *self.0.lock())
    }
}

/// Fusion input for autotuning. Thread the caller's `&mut Context` through the tuning
/// pipeline without `unsafe` by riding cubecl's `'a` lifetime parameter.
pub(crate) struct TuneInput<'a, R: Runtime, O> {
    optimization: Arc<O>,
    state: TuneState<'a, R>,
}

enum TuneState<'a, R: Runtime> {
    Original {
        context: &'a mut Context<CubeFusionHandle<R>>,
        /// Shared with any `Fork` spawned from this `Original`. `Drop` drains from here
        /// if the cache-hit path never ran (e.g. wasm fallback succeeded on a fork).
        new_handles: Arc<HandleCollector<R>>,
        /// Set by [`TuneInput::execute`] when the winner ran on this context, which
        /// suppresses the drain above.
        executed: bool,
    },
    /// Owned fork. `new_handles = None` is a benchmark-only sandbox (outputs discarded);
    /// `Some(…)` is the wasm try-all path that promotes outputs back into the paired
    /// `Original` on drop.
    Fork {
        context: Box<Context<CubeFusionHandle<R>>>,
        new_handles: Option<Arc<HandleCollector<R>>>,
    },
}

impl<'a, R: Runtime, O> TuneInput<'a, R, O> {
    pub(crate) fn new(context: &'a mut Context<CubeFusionHandle<R>>, optimization: O) -> Self {
        Self {
            optimization: Arc::new(optimization),
            state: TuneState::Original {
                context,
                new_handles: Arc::new(HandleCollector::new()),
                executed: false,
            },
        }
    }

    /// Fork the context into a non-tracking `Fork` for a benchmark trial. See
    /// [`FusionInputGen`].
    fn for_benchmark(&self) -> Self {
        Self {
            optimization: self.optimization.clone(),
            state: TuneState::Fork {
                context: Box::new(self.context().fork()),
                new_handles: None,
            },
        }
    }

    /// Fork the context with independent GPU allocations for an autotune correctness check.
    #[cfg(feature = "autotune-checks")]
    fn for_check(&self) -> Self {
        let mut context = self.context().fork();
        let ids = context.handles.handle_ids().copied().collect::<Vec<_>>();
        let mut allocations: Vec<(cubecl::server::Handle, cubecl::server::Handle)> = Vec::new();
        let mut handles = Vec::with_capacity(ids.len());

        for id in ids {
            let handle = context.handles.get_handle_ref(&id).unwrap();
            let allocation = match allocations
                .iter()
                .find(|(source, _)| source.is_alias_of(&handle.handle))
            {
                Some((_, allocation)) => allocation.clone(),
                None => {
                    let allocation = clone_allocation_for_check(handle);
                    allocations.push((handle.handle.clone(), allocation.clone()));
                    allocation
                }
            };
            handles.push((id, clone_handle_for_check(handle, allocation)));
        }

        for (id, handle) in handles {
            context.handles.register_handle(id, handle);
        }

        Self {
            optimization: self.optimization.clone(),
            state: TuneState::Fork {
                context: Box::new(context),
                new_handles: None,
            },
        }
    }

    pub(crate) fn is_original(&self) -> bool {
        matches!(self.state, TuneState::Original { .. })
    }

    /// Read-only access to the wrapped context.
    pub(crate) fn context(&self) -> &Context<CubeFusionHandle<R>> {
        match &self.state {
            TuneState::Original { context, .. } => context,
            TuneState::Fork { context, .. } => context,
        }
    }

    pub(crate) fn tensors(&self) -> &HashMap<TensorId, TensorIr> {
        &self.context().tensors
    }

    pub(crate) fn handles(&self) -> &HandleContainer<CubeFusionHandle<R>> {
        &self.context().handles
    }

    pub(crate) fn optimization(&self) -> &O {
        &self.optimization
    }

    /// Consume the input and run `f` with mutable access to the context and
    /// optimization.
    pub(crate) fn execute<F, T>(mut self, f: F) -> T
    where
        F: FnOnce(&mut Context<CubeFusionHandle<R>>, &O) -> T,
    {
        match &mut self.state {
            TuneState::Original {
                context, executed, ..
            } => {
                // Suppresses drop-time persistence; the real context was written directly.
                *executed = true;
                f(context, &self.optimization)
            }
            TuneState::Fork { context, .. } => f(context, &self.optimization),
        }
    }
}

/// Clone a fusion handle onto independent storage while preserving views into its allocation.
#[cfg(feature = "autotune-checks")]
fn clone_allocation_for_check<R: Runtime>(handle: &CubeFusionHandle<R>) -> cubecl::server::Handle {
    use burn_std::{Shape, strides};
    use cubecl::ir::{ElemType, StorageType, UIntKind};
    use cubecl::prelude::TensorBinding;
    use cubecl::std::tensor::into_contiguous;

    let mut source = handle.handle.clone();
    source.offset_start = None;
    source.offset_end = None;
    let size = source.size() as usize;
    let input = TensorBinding {
        handle: source.binding(),
        strides: strides![1],
        shape: Shape::new([size]),
        runtime: PhantomData,
    };
    into_contiguous::<R>(
        &handle.client,
        input,
        StorageType::Scalar(ElemType::UInt(UIntKind::U8)),
    )
    .handle
}

#[cfg(feature = "autotune-checks")]
fn clone_handle_for_check<R: Runtime>(
    handle: &CubeFusionHandle<R>,
    mut copied: cubecl::server::Handle,
) -> CubeFusionHandle<R> {
    copied.offset_start = handle.handle.offset_start;
    copied.offset_end = handle.handle.offset_end;

    CubeFusionHandle {
        client: handle.client.clone(),
        handle: copied,
        device: handle.device.clone(),
        dtype: handle.dtype,
        strides: handle.strides.clone(),
        qparams: handle.qparams.clone(),
    }
}

impl<'a, R: Runtime, O> Clone for TuneInput<'a, R, O> {
    fn clone(&self) -> Self {
        // `Original` clones come from the wasm fallback path (`operations.fastest(i)
        // .execute(inputs.clone())`) and must track outputs. `Fork` clones inherit the
        // source's tracking (benchmark forks stay non-tracking).
        let new_handles = match &self.state {
            TuneState::Original { new_handles, .. } => Some(new_handles.clone()),
            TuneState::Fork { new_handles, .. } => new_handles.clone(),
        };
        Self {
            optimization: self.optimization.clone(),
            state: TuneState::Fork {
                context: Box::new(self.context().fork()),
                new_handles,
            },
        }
    }
}

#[cfg(all(test, feature = "autotune-checks"))]
mod tests {
    use super::TuneInput;
    use crate::CubeFusionHandle;
    use burn_fusion::stream::Context;
    use burn_ir::{HandleContainer, TensorId};
    use burn_std::{DType, strides};
    use cubecl::{Runtime, TestRuntime};
    use hashbrown::HashMap;

    #[test]
    fn test_for_check_isolates_handle_allocations() {
        let device = Default::default();
        let client = TestRuntime::client(&device);
        let id = TensorId::new(0);
        let mut context = Context {
            tensors: HashMap::new(),
            handles: HandleContainer::new(),
            scalars: HashMap::new(),
            shapes_relative2global: HashMap::new(),
        };
        context.handles.register_handle(
            id,
            CubeFusionHandle {
                client: client.clone(),
                handle: client.create_from_slice(&[1, 2, 3, 4]),
                device,
                dtype: DType::F32,
                strides: strides![1],
                qparams: None,
            },
        );

        let input = TuneInput::new(&mut context, ());
        let check = input.for_check();

        assert!(
            input
                .handles()
                .get_handle_ref(&id)
                .unwrap()
                .handle
                .can_mut(),
            "check fork should not retain the original allocation"
        );
        assert!(
            check
                .handles()
                .get_handle_ref(&id)
                .unwrap()
                .handle
                .can_mut(),
            "check fork should own an independent allocation"
        );

        let _check_clone = check.handles().get_handle_ref(&id).unwrap().handle.clone();
        assert!(
            input
                .handles()
                .get_handle_ref(&id)
                .unwrap()
                .handle
                .can_mut(),
            "cloning the check allocation should not affect the original allocation"
        );
    }

    #[test]
    fn test_for_check_preserves_handle_aliases() {
        let device = Default::default();
        let client = TestRuntime::client(&device);
        let base_id = TensorId::new(0);
        let view_id = TensorId::new(1);
        let base = client.create_from_slice(&[1, 2, 3, 4]);
        let mut view = base.clone();
        view.offset_start = Some(4);
        view.offset_end = Some(4);
        let mut context = Context {
            tensors: HashMap::new(),
            handles: HandleContainer::new(),
            scalars: HashMap::new(),
            shapes_relative2global: HashMap::new(),
        };
        context.handles.register_handle(
            base_id,
            CubeFusionHandle {
                client: client.clone(),
                handle: base,
                device: device.clone(),
                dtype: DType::F32,
                strides: strides![1],
                qparams: None,
            },
        );
        context.handles.register_handle(
            view_id,
            CubeFusionHandle {
                client,
                handle: view,
                device,
                dtype: DType::F32,
                strides: strides![2],
                qparams: None,
            },
        );

        let input = TuneInput::new(&mut context, ());
        let check = input.for_check();
        let original_base = input.handles().get_handle_ref(&base_id).unwrap();
        let check_base = check.handles().get_handle_ref(&base_id).unwrap();
        let check_view = check.handles().get_handle_ref(&view_id).unwrap();

        assert!(
            check_base.handle.is_alias_of(&check_view.handle),
            "check fork should preserve aliases"
        );
        assert!(
            !original_base.handle.is_alias_of(&check_base.handle),
            "check fork should not retain the original allocation"
        );
        assert_eq!(check_view.handle.offset_start, Some(4));
        assert_eq!(check_view.handle.offset_end, Some(4));
        assert_eq!(check_view.strides, strides![2]);
    }
}

impl<'a, R: Runtime, O> Drop for TuneInput<'a, R, O> {
    fn drop(&mut self) {
        match &mut self.state {
            TuneState::Original {
                context,
                new_handles,
                executed,
            } => {
                if *executed {
                    return;
                }
                // Cache-hit path never ran. Drain anything the most recent `Fork`
                // deposited and register entries the real context doesn't already have
                // (those are genuine new outputs, not inputs inherited via `fork`).
                for (id, handle) in new_handles.take() {
                    if context.handles.get_handle_ref(&id).is_none() {
                        context.handles.register_handle(id, handle);
                    }
                }
            }
            TuneState::Fork {
                context,
                new_handles,
            } => {
                if let Some(collector) = new_handles {
                    collector.capture(&context.handles);
                }
            }
        }
    }
}
