use crate::register_once_signal_from_system;

use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{prelude::*, system::SystemId};
use bevy_hierarchy::BuildChildren;
use bevy_log::prelude::*;
use bevy_reflect::{FromReflect, GetTypeRegistration, PartialReflect, Typed}; // Removed unused Reflect
use std::{
    collections::{HashMap, HashSet},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

/// Constant used in signal systems to indicate that the signal chain should terminate
/// at this point for the current execution. It's equivalent to returning `None`.
/// When a system returns `None`, the [`SignalPropagator`] stops traversing down that
/// specific branch of the system graph for the current frame.
///
/// ```
/// use jonmo::TERMINATE;
/// assert_eq!(TERMINATE, None::<()>);
/// ```
pub const TERMINATE: Option<()> = None;

/// A system that can be used with [`SignalExt::map`] to prevent propagation
/// if the incoming value is the same as the previous one.
///
/// Requires the value type `T` to implement `PartialEq`, `Clone`, `Send`, `Sync`, and `'static`.
/// It uses [`Local`] state to store the previous value.
///
/// ```
/// use bevy::prelude::*;
/// use jonmo::dedupe;
///
/// #[derive(Clone, PartialEq)]
/// struct MyValue(i32);
///
/// let mut world = World::new();
/// let mut system = IntoSystem::into_system(dedupe::<MyValue>);
/// system.initialize(&mut world);
///
/// // First run: No previous value, should pass through
/// let result1 = system.run(MyValue(10), &mut world);
/// assert_eq!(result1, Some(MyValue(10)));
///
/// // Second run: Same value, should return None
/// let result2 = system.run(MyValue(10), &mut world);
/// assert_eq!(result2, None);
///
/// // Third run: Different value, should pass through
/// let result3 = system.run(MyValue(20), &mut world);
/// assert_eq!(result3, Some(MyValue(20)));
///
/// // Fourth run: Same as third, should return None
/// let result4 = system.run(MyValue(20), &mut world);
/// assert_eq!(result4, None);
/// ```
pub fn dedupe<T>(In(current): In<T>, mut cache: Local<Option<T>>) -> Option<T>
where
    T: PartialEq + Clone + Send + Sync + 'static,
{
    let mut changed = false;
    if let Some(ref p) = *cache {
        if *p != current {
            changed = true;
        }
    } else {
        changed = true;
    }

    if changed {
        *cache = Some(current.clone());
        Some(current)
    } else {
        None
    }
}

#[derive(Clone, Copy, Deref, Debug)]
pub struct SignalSystem(Entity);

impl<I: 'static, O> From<SystemId<In<I>, O>> for SignalSystem {
    fn from(system_id: SystemId<In<I>, O>) -> Self {
        SignalSystem(system_id.entity())
    }
}

/// Component storing metadata for signal system nodes, primarily for reference counting.
#[derive(Component)]
pub(crate) struct SignalReferenceCount(i32);

impl SignalReferenceCount {
    /// Creates metadata with an initial reference count of 1.
    pub(crate) fn new() -> Self {
        Self(1)
    }

    pub(crate) fn increment(&mut self) {
        self.0 += 1;
    }

    pub(crate) fn decrement(&mut self) -> i32 {
        self.0 -= 1;
        self.0
    }
}

/// Helper to register a system, add the [`SystemRunner`] component, and manage [`SignalNodeMetadata`].
///
/// Ensures the system is registered, attaches a runner component, and handles the
/// reference counting via `SignalNodeMetadata`. Returns the `SystemId`.
pub fn register_signal<I, O, IOO, S, M>(world: &mut World, system: S) -> SignalSystem
where
    I: FromReflect + Send + Sync + 'static,
    O: FromReflect + Send + Sync + 'static,
    IOO: Into<Option<O>> + Send + Sync + 'static,
    S: IntoSystem<In<I>, IOO, M> + Send + Sync + 'static,
    M: Send + Sync + 'static,
{
    register_once_signal_from_system(system).get(world)
}

// pub(crate) fn remove_signal_handles(world: &mut World) {
//     let remove_signal_handles = world
//         .resource_mut::<Events<RemoveSignalHandles>>()
//         .drain()
//         .collect::<Vec<_>>();
//     if !remove_signal_handles.is_empty() {
//         for RemoveSignalHandles(handles) in remove_signal_handles {
//             for handle in handles {
//                 handle.cleanup(world);
//             }
//         }
//     }
// }

/// Helper to connect two system entities in the [`SignalPropagator`] graph.
///
/// Establishes a parent-child relationship, indicating that the output of the `source`
/// system should be passed as input to the `target` system during propagation.
pub fn pipe_signal(world: &mut World, source: SignalSystem, target: SignalSystem) {
    if let Ok(mut parent) = world.get_entity_mut(*source) {
        parent.add_child(*target);
    }
}

/// Helper to register the systems needed for combining two signal branches.
///
/// Creates wrapper systems and a final combining system, managing reference counts
/// and connecting them in the [`SignalPropagator`]. Returns the `SystemId` of the
/// final combiner system and a Vec of all system entities involved in this combine step.
// pub fn combine_signal<OLeft, ORight>(
//     world: &mut World,
//     left_last_entity: SignalSystemId,
//     right_last_entity: SignalSystemId,
// ) -> (
//     SystemId<In<(Option<OLeft>, Option<ORight>)>, Option<(OLeft, ORight)>>,
//     Vec<SignalSystemId>,
// )
// where
//     OLeft: FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static,
//     ORight: FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static,
// {
//     let left_wrapper_id = register_signal::<OLeft, (Option<OLeft>, Option<ORight>), _, _>(
//         world,
//         |In(left_val): In<OLeft>| Some((Some(left_val), None::<ORight>)),
//     );
//     pipe_signal(world, left_last_entity, left_wrapper_id.entity());

//     let right_wrapper_id = register_signal::<ORight, (Option<OLeft>, Option<ORight>), _, _>(
//         world,
//         |In(right_val): In<ORight>| Some((None::<OLeft>, Some(right_val))),
//     );
//     pipe_signal(world, right_last_entity, right_wrapper_id.entity());

//     let combine_id = register_signal::<_, Option<(OLeft, ORight)>, _, _>(
//         world,
//         move |In((left_opt, right_opt)): In<(Option<OLeft>, Option<ORight>)>,
//               mut left_cache: Local<Option<OLeft>>,
//               mut right_cache: Local<Option<ORight>>| {
//             if left_opt.is_some() {
//                 *left_cache = left_opt;
//             }
//             if right_opt.is_some() {
//                 *right_cache = right_opt;
//             }
//             if left_cache.is_some() && right_cache.is_some() {
//                 left_cache.take().zip(right_cache.take())
//             } else {
//                 None
//             }
//         },
//     );

//     pipe_signal(world, left_wrapper_id.entity(), combine_id.entity());
//     pipe_signal(world, right_wrapper_id.entity(), combine_id.entity());

//     let registered_entities = vec![
//         left_wrapper_id.entity(),
//         right_wrapper_id.entity(),
//         combine_id.entity(),
//     ];

//     (combine_id, registered_entities)
// }

/// Component holding the type-erased system runner function.
///
/// This component is attached to the entity associated with each registered signal system.
/// It contains an `Arc<Box<dyn Fn(...)>>` that captures the specific `SystemId` and
/// handles the type-erased execution logic, including downcasting inputs and boxing outputs.
#[derive(Component, Clone)]
pub(crate) struct SystemRunner {
    /// The type-erased function to execute the system.
    pub(crate) runner: Arc<
        Box<
            dyn Fn(&mut World, Box<dyn PartialReflect>) -> Option<Box<dyn PartialReflect>>
                + Send
                + Sync
                + 'static,
        >,
    >,
}

impl SystemRunner {
    /// Executes the stored system function with the given type-erased input.
    ///
    /// Takes the `World` and a `Box<dyn PartialReflect>` input, runs the system,
    /// and returns an optional `Box<dyn PartialReflect>` output.
    pub(crate) fn run(
        &self,
        world: &mut World,
        input: Box<dyn PartialReflect>,
    ) -> Option<Box<dyn PartialReflect>> {
        (self.runner)(world, input)
    }
}
