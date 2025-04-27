use crate::{register_once_signal_from_system, utils::SSs};

use bevy_derive::Deref;
use bevy_ecs::{prelude::*, system::SystemId};
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
pub fn register_signal<I, O, IOO, IS, M>(world: &mut World, system: IS) -> SignalSystem
where
    I: FromReflect + SSs,
    O: FromReflect + SSs,
    IOO: Into<Option<O>> + SSs,
    IS: IntoSystem<In<I>, IOO, M> + SSs,
    M: SSs,
{
    register_once_signal_from_system(system).register(world)
}

// TODO: 0.16 relationships
#[derive(Component, Deref)]
pub(crate) struct Upstream(HashSet<SignalSystem>);

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
