//! Declarative entity builder using jonmo signals.

use crate::{
    prelude::*,
    signal::{Signal, /* SignalBuilderInternal, */ SignalExt, SignalHandle},
};
use bevy_ecs::{prelude::*, system::SystemState};
use bevy_hierarchy::prelude::*;
use bevy_reflect::{/* FromReflect, */ FromReflect, Reflect};
use std::sync::{Arc, Mutex};

/// Component storing handles to reactive systems attached to an entity.
/// These handles are used to clean up the systems when the entity is despawned.
#[derive(Component, Default)]
pub struct SignalHandlers {
    handles: Vec<SignalHandle>,
}

impl SignalHandlers {
    /// Add a signal handle to the component.
    pub fn add(&mut self, handle: SignalHandle) {
        self.handles.push(handle);
    }
}

/// A thin facade over a Bevy [`Entity`] enabling the ergonomic registration of reactive systems and
/// children using a declarative builder pattern. Inspired by Dominator's DomBuilder and Haalka's NodeBuilder.
#[derive(Default, Clone, Reflect)] // Removed Clone
#[reflect(opaque)]
pub struct NodeBuilder {
    #[allow(clippy::type_complexity)]
    on_spawns: Arc<Mutex<Vec<Box<dyn FnOnce(&mut World, Entity) + Send + Sync>>>>, // Changed type
    child_block_populations: Arc<Mutex<Vec<usize>>>, // Keep this for child logic for now
}

impl<T: Bundle> From<T> for NodeBuilder {
    fn from(bundle: T) -> Self {
        Self::new().insert(bundle)
    }
}

impl NodeBuilder {
    /// Create a new, empty [`NodeBuilder`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Run a function with mutable access to the [`World`] and this node's [`Entity`]
    /// immediately after the entity is spawned, but before signal systems are fully registered.
    pub fn on_spawn(
        self, // Remove mut
        on_spawn: impl FnOnce(&mut World, Entity) + Send + Sync + 'static,
    ) -> Self {
        self.on_spawns.lock().unwrap().push(Box::new(on_spawn)); // Direct access
        self
    }

    /// Insert a [`Bundle`] onto the node's entity.
    pub fn insert<T: Bundle>(self, bundle: T) -> Self {
        self.on_spawn(move |world, entity| {
            if let Ok(mut entity) = world.get_entity_mut(entity) {
                entity.insert(bundle);
            }
        })
    }

    /// Register a reactive system that runs when the given [`Signal`] emits a value.
    /// The system receives the value emitted by the signal.
    /// Note: The system is registered *directly* on the builder and will be transferred
    /// to the entity's `SignalHandlers` component upon spawning.
    // pub fn on_signal<S, Marker>(mut self, signal: S, system: impl IntoSystem<S::Item, (), Marker>) -> Self
    // where
    //     S: Signal + Send + Sync + 'static,
    //     S::Item: Send + Sync + 'static,
    // {
    //     let handle = signal.into_signal_builder().create_system(system);
    //     self.signal_handlers.push(handle); // Add handle directly to builder's vec
    //     self
    // }

    /// Register a reactive system that runs when the given [`Signal`] emits a value.
    /// The system receives the value emitted by the signal and the node's [`Entity`].
    /// Note: This registers an `on_spawn` callback. The system itself is created
    /// and added to the entity's `SignalHandlers` component *during* the `on_spawn` execution.
    // pub fn on_signal_with_entity<S, Marker>(
    //     self,
    //     signal: S,
    //     system: impl IntoSystem<(S::Item, Entity), (), Marker>,
    // ) -> Self
    // where
    //     S: Signal + Send + Sync + 'static,
    //     S::Item: Send + Sync + 'static,
    // {
    //     self.on_spawn(move |world, entity| {
    //         // Create the system handle inside the on_spawn callback
    //         let handle = signal
    //             .into_signal_builder()
    //             .create_system_with_entity(entity, system);

    //         // Add the handle to the entity's SignalHandlers component
    //         if let Some(mut handlers) = world.get_mut::<SignalHandlers>(entity) {
    //             handlers.add(handle);
    //         } else {
    //              warn!("Entity {:?} despawned or missing SignalHandlers before on_signal_with_entity could register handler.", entity);
    //              // If the entity or component is gone, we need to clean up the handle we just created.
    //              handle.cleanup(world);
    //         }
    //     })
    // }

    /// Declare a static child node.
    /// The child is spawned and added to the parent when the parent is spawned.
    pub fn child(self, child: NodeBuilder) -> Self {
        let block = self.child_block_populations.lock().unwrap().len();
        self.child_block_populations.lock().unwrap().push(1);
        let offset = offset(block, &self.child_block_populations.lock().unwrap());
        let on_spawn = move |world: &mut World, parent| {
            let child_entity = world.spawn_empty().id();
            if let Ok(ref mut parent) = world.get_entity_mut(parent) {
                EntityWorldMut::insert_children(parent, offset, &[child_entity]);
                child.spawn_on_entity(world, child_entity);
            } else {
                if let Ok(child) = world.get_entity_mut(child_entity) {
                    child.despawn_recursive();
                }
            }
        };
        self.on_spawn(on_spawn)
    }

    /// Declare a reactive child. When the [`Signal`] outputs [`None`], the child is removed.
    pub fn child_signal<T: Into<Option<NodeBuilder>> + FromReflect>(
        self,
        child_option: impl Signal<Item = T> + Clone + Send + Sync + 'static,
    ) -> Self {
        let block = self.child_block_populations.lock().unwrap().len();
        self.child_block_populations.lock().unwrap().push(0);
        let child_block_populations = self.child_block_populations.clone();
        let on_spawn = move |world: &mut World, parent: Entity| {
            let system = move |In(child_option): In<T>, world: &mut World| {
                let mut existing_child_option: SystemState<Local<Option<Entity>>> = SystemState::new(world);
                if let Some(child) = child_option.into() {
                    if let Some(existing_child) = *existing_child_option.get(world) {
                        if let Ok(entity) = world.get_entity_mut(existing_child) {
                            entity.despawn_recursive();
                        }
                    }
                    let child_entity = world.spawn_empty().id();
                    if let Ok(mut parent) = world.get_entity_mut(parent) {
                        let offset = offset(block, &child_block_populations.lock().unwrap());
                        parent.insert_children(offset, &[child_entity]);
                        child.spawn_on_entity(world, child_entity);
                        *existing_child_option.get_mut(world) = Some(child_entity);
                    } else {
                        if let Ok(child) = world.get_entity_mut(child_entity) {
                            child.despawn_recursive();
                        }
                    }
                    child_block_populations.lock().unwrap()[block] = 1;
                } else {
                    if let Some(existing_child) = existing_child_option.get_mut(world).take() {
                        if let Ok(entity) = world.get_entity_mut(existing_child) {
                            entity.despawn_recursive();
                        }
                    }
                    child_block_populations.lock().unwrap()[block] = 0;
                }
                TERMINATE
            };
            let signal = child_option.map(system);
            let handle = Signal::register(&signal, world);
            if let Ok(mut entity) = world.get_entity_mut(parent) {
                if let Some(mut handlers) = entity.get_mut::<SignalHandlers>() {
                    handlers.add(handle);
                }
            }
        };
        self.on_spawn(on_spawn)
    }

    /// Declare static children.
    pub fn children(
        self,
        children: impl IntoIterator<Item = NodeBuilder> + Send + 'static,
    ) -> Self {
        let block = self.child_block_populations.lock().unwrap().len();
        let children_vec: Vec<NodeBuilder> = children.into_iter().collect(); // Collect into Vec
        let population = children_vec.len();
        self.child_block_populations
            .lock()
            .unwrap()
            .push(population);
        let child_block_populations = self.child_block_populations.clone(); // Clone Arc

        let on_spawn = move |world: &mut World, parent: Entity| {
            let mut children_entities = Vec::with_capacity(children_vec.len());
            for _ in 0..children_vec.len() {
                children_entities.push(world.spawn_empty().id());
            }

            if let Ok(mut parent) = world.get_entity_mut(parent) {
                let offset = offset(block, &child_block_populations.lock().unwrap()); // Recalculate offset
                parent.insert_children(offset, &children_entities);
                for (child, child_entity) in children_vec
                    .into_iter()
                    .zip(children_entities.iter().copied())
                {
                    // Use copied iterator
                    child.spawn_on_entity(world, child_entity);
                }
            } else {
                // Parent despawned during child spawning
                for child_entity in children_entities {
                    if let Ok(child) = world.get_entity_mut(child_entity) {
                        child.despawn_recursive();
                    }
                }
            }
        };
        self.on_spawn(on_spawn)
    }

    /// Declare reactive children based on a `SignalVec`.
    pub fn children_signal_vec(
        self,
        children_signal_vec: impl SignalVec<Item = NodeBuilder> + Clone,
    ) -> Self {
        let block = self.child_block_populations.lock().unwrap().len();
        self.child_block_populations.lock().unwrap().push(0); // Initial population is 0
        let child_block_populations = self.child_block_populations.clone();

        let on_spawn = move |world: &mut World, parent: Entity| {
            // Define the system that will handle VecDiff updates
            let system = move |In(diffs): In<Vec<VecDiff<NodeBuilder>>>, world: &mut World| {
                let mut children_entities: SystemState<Local<Vec<Entity>>> = SystemState::new(world);
                for diff in diffs {
                    match diff {
                        VecDiff::Replace { values: children } => {
                            for child_entity in children_entities.get_mut(world).drain(..) {
                                if let Ok(child) = world.get_entity_mut(child_entity) {
                                    child.despawn_recursive();
                                }
                            }
                            *children_entities.get_mut(world) =
                                children.iter().map(|_| world.spawn_empty().id()).collect();
                            if let Ok(mut parent) = world.get_entity_mut(parent) {
                                let offset =
                                    offset(block, &child_block_populations.lock().unwrap());
                                parent.insert_children(offset, &children_entities.get(world));
                                for (child, child_entity) in children
                                    .into_iter()
                                    .zip(children_entities.get(world).iter().copied())
                                {
                                    child.spawn_on_entity(world, child_entity);
                                }
                                child_block_populations.lock().unwrap()[block] =
                                    children_entities.get(world).len();
                            }
                        }
                        VecDiff::InsertAt { index, value: child } => {
                            let child_entity = world.spawn_empty().id();
                            if let Ok(mut parent) = world.get_entity_mut(parent) {
                                let offset = offset(
                                    block,
                                    &child_block_populations.lock().unwrap(),
                                );
                                parent.insert_children(offset + index, &[child_entity])
                                child.spawn_on_entity(world, child_entity);
                                children_entities.get_mut(world).insert(index, child_entity);
                                child_block_populations.lock().unwrap()[block] = children_entities.get(world).len();
                            } else {
                                // Parent despawned during child insertion
                                if let Ok(child) = world.get_entity_mut(child_entity) {
                                    child.despawn_recursive();
                                }
                            }
                        }
                        VecDiff::Push { value: child } => {
                            let child_entity = world.spawn_empty().id();
                            if let Ok(mut parent) = world.get_entity_mut(parent) {
                                let offset = offset(block, &child_block_populations.lock().unwrap());
                                parent.insert_children(offset + children_entities.get(world).len(), &[child_entity]);
                                child.spawn_on_entity(world, child_entity);
                                children_entities.get_mut(world).push(child_entity);
                                child_block_populations.lock().unwrap()[block] = children_entities.get(world).len();
                            } else {  // parent despawned during child spawning
                                if let Ok(child) = world.get_entity_mut(child_entity) {
                                    child.despawn_recursive();
                                }
                            }
                        }
                        VecDiff::UpdateAt { index, value: node } => {
                            if let Some(existing_child) = children_entities.get(world).get(index).copied() {
                                if let Ok(child) = world.get_entity_mut(existing_child) {
                                    child.despawn_recursive();  // removes from parent
                                }
                            }
                            let child_entity = world.spawn_empty().id();
                            if let Ok(mut parent) = world.get_entity_mut(parent) {
                                children_entities.get_mut(world)[index] = child_entity;
                                let offset = offset(block, &child_block_populations.lock().unwrap());
                                parent.insert_children(offset + index, &[child_entity]);
                                node.spawn_on_entity(world, child_entity);
                            } else {  // parent despawned during child spawning
                                if let Ok(child) = world.get_entity_mut(child_entity) {
                                    child.despawn_recursive();
                                }
                            }
                        }
                        VecDiff::Move {
                            old_index,
                            new_index,
                        } => {
                            children_entities.get_mut(world).swap(old_index, new_index);
                            fn move_from_to(parent: &mut EntityWorldMut, children_entities: &[Entity], old_index: usize, new_index: usize) {
                                if old_index != new_index {
                                    if let Some(old_entity) = children_entities.get(old_index).copied() {
                                        parent.remove_children(&[old_entity]);
                                        parent.insert_children(new_index, &[old_entity]);
                                    }
                                }
                            }
                            fn swap(parent: &mut EntityWorldMut, children_entities: &[Entity], a: usize, b: usize) {
                                move_from_to(parent, children_entities, a, b);
                                match a.cmp(&b) {
                                    std::cmp::Ordering::Less => {
                                        move_from_to(parent, children_entities, b - 1, a);
                                    }
                                    std::cmp::Ordering::Greater => {
                                        move_from_to(parent, children_entities, b + 1, a)
                                    }
                                    _ => {}
                                }
                            }
                            if let Ok(mut parent) = world.get_entity_mut(parent) {
                                let offset = offset(block, &child_block_populations.lock().unwrap());
                                swap(&mut parent, children_entities.get(world).as_slice(), offset + old_index, offset + new_index);
                            }
                        }
                        VecDiff::RemoveAt { index } => {
                            if let Some(existing_child) = children_entities.get(world).get(index).copied() {
                                if let Ok(child) = world.get_entity_mut(existing_child) {
                                    child.despawn_recursive();  // removes from parent
                                }
                                children_entities.get_mut(world).remove(index);
                                child_block_populations.lock().unwrap()[block] = children_entities.get(world).len();
                            }
                        }
                        VecDiff::Pop => {
                            if let Some(child_entity) = children_entities.get_mut(world).pop() {
                                if let Ok(child) = world.get_entity_mut(child_entity) {
                                    child.despawn_recursive();
                                }
                                child_block_populations.lock().unwrap()[block] = children_entities.get(world).len();
                            }
                        }
                        VecDiff::Clear => {
                            for child_entity in children_entities.get_mut(world).drain(..) {
                                if let Ok(child) = world.get_entity_mut(child_entity) {
                                    child.despawn_recursive();
                                }
                            }
                            child_block_populations.lock().unwrap()[block] = children_entities.get(world).len();
                        }
                    }
                }
                TERMINATE
            };
            let signal = children_signal_vec.map(system);
            let handle = SignalVec::register(&signal, world);

            if let Ok(mut entity_mut) = world.get_entity_mut(parent) {
                if let Some(mut handlers) = entity_mut.get_mut::<SignalHandlers>() {
                    handlers.add(handle);
                }
            }
        };

        self.on_spawn(on_spawn)
    }

    /// Spawn the node with its components and reactive systems onto an existing [`Entity`].
    ///
    /// Note: This consumes the builder.
    pub fn spawn_on_entity(self, world: &mut World, entity: Entity) {
        if let Ok(mut entity) = world.get_entity_mut(entity) {
            let id = entity.id();
            entity.insert(SignalHandlers::default());
            for on_spawn in self.on_spawns.lock().unwrap().drain(..) {
                on_spawn(world, id);
            }
        }
    }

    /// Spawn the node with its components and reactive systems onto a new [`Entity`].
    pub fn spawn(self, world: &mut World) -> Entity {
        let entity = world.spawn_empty().id();
        self.spawn_on_entity(world, entity);
        entity
    }
}

fn offset(i: usize, child_block_populations: &[usize]) -> usize {
    child_block_populations[0..i].iter().copied().sum()
}
