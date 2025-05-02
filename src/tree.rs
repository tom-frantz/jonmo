use crate::{register_once_signal_from_system, utils::SSs};

use bevy_derive::Deref;
use bevy_ecs::{
    component::ComponentId,
    prelude::*,
    system::{RunSystemOnce, SystemId, SystemState},
    world::DeferredWorld,
};
use bevy_reflect::{FromReflect, PartialReflect, Reflect};
use std::{collections::HashSet, hash::Hash, sync::Arc};

#[derive(Clone, Copy, Deref, Debug, PartialEq, Eq, Hash, Reflect)]
pub struct SignalSystem(pub Entity);

impl<I: 'static, O> From<SystemId<In<I>, O>> for SignalSystem {
    fn from(system_id: SystemId<In<I>, O>) -> Self {
        SignalSystem(system_id.entity())
    }
}

impl SignalSystem {
    /// Returns the underlying [`Entity`] of this signal system.
    pub fn entity(&self) -> Entity {
        self.0
    }
}

/// Component storing metadata for signal system nodes, primarily for reference counting.
#[derive(Component)]
pub(crate) struct SignalRegistrationCount(i32);

impl SignalRegistrationCount {
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
pub fn register_signal<I, O, IOO, F, M>(world: &mut World, system: F) -> SignalSystem
where
    I: FromReflect + SSs,
    O: FromReflect + SSs,
    IOO: Into<Option<O>> + SSs,
    F: IntoSystem<In<I>, IOO, M> + SSs,
    M: SSs,
{
    register_once_signal_from_system(system).register(world)
}

fn downstream_syncer(mut world: DeferredWorld, entity: Entity, _: ComponentId) {
    world.commands().queue(move |world: &mut World| {
        let _ = world.run_system_once(
            move |upstreams: Query<&Upstream>,
                  mut downstreams: Query<&mut Downstream>,
                  mut commands: Commands| {
                if let Ok(upstream) = upstreams.get(entity) {
                    for &upstream_system in upstream.iter() {
                        if let Ok(mut downstreams) = downstreams.get_mut(*upstream_system) {
                            downstreams.0.remove(&SignalSystem(entity));
                            if downstreams.0.is_empty() {
                                if let Some(mut entity) = commands.get_entity(*upstream_system) {
                                    entity.remove::<Downstream>();
                                }
                            }
                        }
                    }
                }
            },
        );
    });
}

// TODO: 0.16 relationships
#[derive(Component, Deref, Clone)]
#[component(on_remove = downstream_syncer)]
pub(crate) struct Upstream(pub(crate) HashSet<SignalSystem>);

impl<'a> IntoIterator for &'a Upstream {
    type Item = <Self::IntoIter as Iterator>::Item;

    type IntoIter = std::collections::hash_set::Iter<'a, SignalSystem>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

#[derive(Component, Deref)]
pub(crate) struct Downstream(HashSet<SignalSystem>);

impl<'a> IntoIterator for &'a Downstream {
    type Item = <Self::IntoIter as Iterator>::Item;

    type IntoIter = std::collections::hash_set::Iter<'a, SignalSystem>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

pub fn pipe_signal(world: &mut World, source: SignalSystem, target: SignalSystem) {
    if let Ok(mut upstream) = world.get_entity_mut(*source) {
        if let Some(mut downstream) = upstream.get_mut::<Downstream>() {
            downstream.0.insert(target);
        } else {
            upstream.insert(Downstream(HashSet::from([target])));
        }
    }
    if let Ok(mut downstream) = world.get_entity_mut(*target) {
        if let Some(mut upstream) = downstream.get_mut::<Upstream>() {
            upstream.0.insert(source);
        } else {
            downstream.insert(Upstream(HashSet::from([source])));
        }
    }
}

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

fn clone_downstream(downstream: &Downstream) -> Vec<SignalSystem> {
    downstream.iter().cloned().collect()
}

pub(crate) fn process_signals_helper(
    world: &mut World,
    signals: impl IntoIterator<Item = SignalSystem>,
    input: Box<dyn PartialReflect>,
) {
    for signal in signals {
        if let Some(runner) = world
            .get_entity(*signal)
            .ok()
            .and_then(|entity| entity.get::<SystemRunner>().cloned())
        {
            if let Some(output) = runner.run(world, input.clone_value()) {
                if let Some(downstream) = world.get::<Downstream>(*signal).map(clone_downstream) {
                    process_signals_helper(world, downstream, output);
                }
            }
        }
    }
}

/// System that drives signal propagation by calling [`SignalPropagator::execute`].
/// Added to the `Update` schedule by the [`JonmoPlugin`]. This system runs once per frame.
/// It temporarily removes the [`SignalPropagator`] resource to allow mutable access to the `World`
/// during system execution within the propagator.
pub(crate) fn process_signals(world: &mut World) {
    let mut orphan_parents = SystemState::<
        Query<Entity, (With<SystemRunner>, Without<Upstream>, With<Downstream>)>,
    >::new(world);
    let orphan_parents = orphan_parents.get(world);
    let orphan_parents = orphan_parents.iter().map(SignalSystem).collect::<Vec<_>>();
    process_signals_helper(world, orphan_parents, Box::new(()));
}
