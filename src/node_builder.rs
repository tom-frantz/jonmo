//! Declarative entity builder using jonmo signals.

use crate::{
    prelude::*,
    register_signal,
    signal::{Signal, SignalExt, SignalHandle},
};
use bevy_ecs::{component::ComponentId, prelude::*, world::DeferredWorld};
use bevy_hierarchy::prelude::*;
use bevy_reflect::{FromReflect, GetTypeRegistration, Reflect, Typed};
use std::sync::{Arc, Mutex, OnceLock};

fn cleanup_signal_handlers(mut world: DeferredWorld, entity: Entity, _: ComponentId) {
    if let Ok(entity_mut) = world.get_entity(entity) {
        if let Some(handles) = entity_mut.get::<SignalHandles>() {
            world.send_event(RemoveSignalHandles(handles.handles.clone()));
        }
    }
}

/// Component storing handles to reactive systems attached to an entity.
/// These handles are used to clean up the systems when the entity is despawned.
#[derive(Component, Default)]
#[component(on_remove = cleanup_signal_handlers)]
pub struct SignalHandles {
    handles: Vec<SignalHandle>,
}

#[derive(Event)]
pub(crate) struct RemoveSignalHandles(pub(crate) Vec<SignalHandle>);

impl SignalHandles {
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

    /// Sync the entity with an [`OnceLock<Entity>`].
    pub fn entity_sync(self, entity: Arc<OnceLock<Entity>>) -> Self {
        self.on_spawn(move |_, e| {
            let _ = entity.set(e);
        })
    }

    /// Register a reactive system that runs when the given [`Signal`] emits a value.
    /// The system receives the value emitted by the signal.
    /// Note: The system is registered *directly* on the builder and will be transferred
    /// to the entity's `SignalHandlers` component upon spawning.
    pub fn on_signal<S, F, I, O, M>(self, signal: S, system: F) -> Self
    where
        S: Signal<Item = I> + Send + Sync + 'static,
        F: IntoSystem<In<(Entity, I)>, Option<O>, M> + Send + Sync + Clone + 'static,
        I: FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static,
        O: FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static,
        M: Send + Sync + 'static,
    {
        let on_spawn = move |world: &mut World, entity: Entity| {
            let system = register_signal(world, system);
            let wrapper_system = move |In(input): In<I>, world: &mut World| {
                world.run_system_with_input(system, (entity, input)).ok()
            };
            let signal = signal.map(wrapper_system);
            let handle = Signal::register(&signal, world);
            if let Ok(mut entity) = world.get_entity_mut(entity) {
                if let Some(mut handlers) = entity.get_mut::<SignalHandles>() {
                    handlers.add(handle);
                }
            }
        };
        self.on_spawn(on_spawn)
    }

    /// Register a reactive system that runs when the given [`Signal`] emits a value.
    pub fn signal_from_entity<OS, F, O>(self, f: F) -> Self
    where
        OS: Signal<Item = O> + Send + Sync + 'static,
        F: FnOnce(Source<Entity>) -> OS + Send + Sync + 'static,
        O: FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static,
    {
        let on_spawn = move |world: &mut World, entity: Entity| {
            // TODO: reuse this entity emitting signal somehow ?
            let signal = f(SignalBuilder::from_entity(entity));
            let handle = Signal::register(&signal, world);
            if let Ok(mut entity) = world.get_entity_mut(entity) {
                if let Some(mut handlers) = entity.get_mut::<SignalHandles>() {
                    handlers.add(handle);
                }
            }
        };
        self.on_spawn(on_spawn)
    }

    /// Register a reactive system that runs when the given [`Signal`] emits a value.
    pub fn signal_from_component<C, OS, F, O>(self, f: F) -> Self
    where
        C: Component + Clone + FromReflect + GetTypeRegistration + Typed,
        OS: Signal<Item = O> + Send + Sync + 'static,
        F: FnOnce(Map<Source<Entity>, C>) -> OS + Send + Sync + 'static,
        O: FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static,
    {
        self.signal_from_entity(|signal| {
            f(signal.map(|In(entity): In<Entity>, components: Query<&C>| {
                components.get(entity).ok().cloned()
            }))
        })
    }

    /// Register a reactive system that runs when the given [`Signal`] emits a value.
    pub fn component_signal<S, O>(self, signal: S) -> Self
    where
        S: Signal<Item = Option<O>> + Send + Sync + 'static,
        O: Component + FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static,
    {
        self.on_signal(
            signal,
            move |In((entity, component_option)): In<(Entity, Option<O>)>, world: &mut World| {
                if let Ok(mut entity) = world.get_entity_mut(entity) {
                    if let Some(component) = component_option {
                        entity.insert(component);
                    } else {
                        entity.remove::<O>();
                    }
                }
                TERMINATE
            },
        )
    }

    /// Register a reactive system that runs when the given [`Signal`] emits a value.
    pub fn component_signal_from_entity<OS, F, O>(self, f: F) -> Self
    where
        OS: Signal<Item = Option<O>> + Send + Sync + 'static,
        F: FnOnce(Source<Entity>) -> OS + Send + Sync + 'static,
        O: Component + FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static,
    {
        let entity = Arc::new(OnceLock::new());
        self.entity_sync(entity.clone())
            .signal_from_entity(move |signal| {
                f(signal).map(move |In(component_option), world: &mut World| {
                    if let Ok(mut entity) = world.get_entity_mut(entity.get().copied().unwrap()) {
                        if let Some(component) = component_option {
                            entity.insert(component);
                        } else {
                            entity.remove::<O>();
                        }
                    }
                    TERMINATE
                })
            })
    }

    /// Register a reactive system that runs when the given [`Signal`] emits a value.
    pub fn component_signal_from_component<C, OS, F, O>(self, f: F) -> Self
    where
        C: Component + Clone + FromReflect + GetTypeRegistration + Typed,
        OS: Signal<Item = Option<O>> + Send + Sync + 'static,
        F: FnOnce(Map<Source<Entity>, C>) -> OS + Send + Sync + 'static,
        O: Component + FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static,
    {
        self.component_signal_from_entity(|signal| {
            f(signal.map(|In(entity): In<Entity>, components: Query<&C>| {
                components.get(entity).ok().cloned()
            }))
        })
    }

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
        child_option: impl Signal<Item = T> + Send + Sync + 'static,
    ) -> Self {
        let block = self.child_block_populations.lock().unwrap().len();
        self.child_block_populations.lock().unwrap().push(0);
        let child_block_populations = self.child_block_populations.clone();
        let on_spawn = move |world: &mut World, parent: Entity| {
            let system =
                move |In(child_option): In<T>,
                      world: &mut World,
                      mut existing_child_option: Local<Option<Entity>>| {
                    if let Some(child) = child_option.into() {
                        if let Some(existing_child) = existing_child_option.take() {
                            if let Ok(entity) = world.get_entity_mut(existing_child) {
                                entity.despawn_recursive();
                            }
                        }
                        let child_entity = world.spawn_empty().id();
                        if let Ok(mut parent) = world.get_entity_mut(parent) {
                            let offset = offset(block, &child_block_populations.lock().unwrap());
                            parent.insert_children(offset, &[child_entity]);
                            child.spawn_on_entity(world, child_entity);
                            *existing_child_option = Some(child_entity);
                        } else {
                            if let Ok(child) = world.get_entity_mut(child_entity) {
                                child.despawn_recursive();
                            }
                        }
                        child_block_populations.lock().unwrap()[block] = 1;
                    } else {
                        if let Some(existing_child) = existing_child_option.take() {
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
                if let Some(mut handlers) = entity.get_mut::<SignalHandles>() {
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
            let system = move |In(diffs): In<Vec<VecDiff<NodeBuilder>>>,
                               world: &mut World,
                               mut children_entities: Local<Vec<Entity>>| {
                for diff in diffs {
                    match diff {
                        VecDiff::Replace { values: children } => {
                            for child_entity in children_entities.drain(..) {
                                if let Ok(child) = world.get_entity_mut(child_entity) {
                                    child.despawn_recursive();
                                }
                            }
                            *children_entities =
                                children.iter().map(|_| world.spawn_empty().id()).collect();
                            if let Ok(mut parent) = world.get_entity_mut(parent) {
                                let offset =
                                    offset(block, &child_block_populations.lock().unwrap());
                                parent.insert_children(offset, &children_entities);
                                for (child, child_entity) in
                                    children.into_iter().zip(children_entities.iter().copied())
                                {
                                    child.spawn_on_entity(world, child_entity);
                                }
                                child_block_populations.lock().unwrap()[block] =
                                    children_entities.len();
                            }
                        }
                        VecDiff::InsertAt {
                            index,
                            value: child,
                        } => {
                            let child_entity = world.spawn_empty().id();
                            if let Ok(mut parent) = world.get_entity_mut(parent) {
                                let offset =
                                    offset(block, &child_block_populations.lock().unwrap());
                                parent.insert_children(offset + index, &[child_entity]);
                                child.spawn_on_entity(world, child_entity);
                                children_entities.insert(index, child_entity);
                                child_block_populations.lock().unwrap()[block] =
                                    children_entities.len();
                            } else {
                                // Parent despawned during child insertion
                                if let Ok(child) = world.get_entity_mut(child_entity) {
                                    child.despawn_recursive();
                                }
                            }
                        }
                        VecDiff::Push { value: child } => {
                            let child_entity = world.spawn_empty().id();
                            let mut push_child_entity = false;
                            {
                                if let Ok(mut parent) = world.get_entity_mut(parent) {
                                    let offset =
                                        offset(block, &child_block_populations.lock().unwrap());
                                    parent.insert_children(
                                        offset + children_entities.len(),
                                        &[child_entity],
                                    );
                                    child.spawn_on_entity(world, child_entity);
                                    push_child_entity = true;
                                    child_block_populations.lock().unwrap()[block] =
                                        children_entities.len();
                                } else {
                                    // parent despawned during child spawning
                                    if let Ok(child) = world.get_entity_mut(child_entity) {
                                        child.despawn_recursive();
                                    }
                                }
                            }
                            if push_child_entity {
                                children_entities.push(child_entity);
                            }
                        }
                        VecDiff::UpdateAt { index, value: node } => {
                            if let Some(existing_child) = children_entities.get(index).copied() {
                                if let Ok(child) = world.get_entity_mut(existing_child) {
                                    child.despawn_recursive(); // removes from parent
                                }
                            }
                            let child_entity = world.spawn_empty().id();
                            let mut set_child_entity = false;
                            if let Ok(mut parent) = world.get_entity_mut(parent) {
                                set_child_entity = true;
                                let offset =
                                    offset(block, &child_block_populations.lock().unwrap());
                                parent.insert_children(offset + index, &[child_entity]);
                                node.spawn_on_entity(world, child_entity);
                            } else {
                                // parent despawned during child spawning
                                if let Ok(child) = world.get_entity_mut(child_entity) {
                                    child.despawn_recursive();
                                }
                            }
                            if set_child_entity {
                                children_entities[index] = child_entity;
                            }
                        }
                        VecDiff::Move {
                            old_index,
                            new_index,
                        } => {
                            children_entities.swap(old_index, new_index);
                            fn move_from_to(
                                parent: &mut EntityWorldMut,
                                children_entities: &[Entity],
                                old_index: usize,
                                new_index: usize,
                            ) {
                                if old_index != new_index {
                                    if let Some(old_entity) =
                                        children_entities.get(old_index).copied()
                                    {
                                        parent.remove_children(&[old_entity]);
                                        parent.insert_children(new_index, &[old_entity]);
                                    }
                                }
                            }
                            fn swap(
                                parent: &mut EntityWorldMut,
                                children_entities: &[Entity],
                                a: usize,
                                b: usize,
                            ) {
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
                                let offset =
                                    offset(block, &child_block_populations.lock().unwrap());
                                swap(
                                    &mut parent,
                                    &children_entities,
                                    offset + old_index,
                                    offset + new_index,
                                );
                            }
                        }
                        VecDiff::RemoveAt { index } => {
                            if let Some(existing_child) = children_entities.get(index).copied() {
                                if let Ok(child) = world.get_entity_mut(existing_child) {
                                    child.despawn_recursive(); // removes from parent
                                }
                                children_entities.remove(index);
                                child_block_populations.lock().unwrap()[block] =
                                    children_entities.len();
                            }
                        }
                        VecDiff::Pop => {
                            if let Some(child_entity) = children_entities.pop() {
                                if let Ok(child) = world.get_entity_mut(child_entity) {
                                    child.despawn_recursive();
                                }
                                child_block_populations.lock().unwrap()[block] =
                                    children_entities.len();
                            }
                        }
                        VecDiff::Clear => {
                            for child_entity in children_entities.drain(..) {
                                if let Ok(child) = world.get_entity_mut(child_entity) {
                                    child.despawn_recursive();
                                }
                            }
                            child_block_populations.lock().unwrap()[block] =
                                children_entities.len();
                        }
                    }
                }
                TERMINATE
            };
            let signal = children_signal_vec.for_each(system);
            let handle = SignalVec::register(&signal, world);

            if let Ok(mut entity_mut) = world.get_entity_mut(parent) {
                if let Some(mut handlers) = entity_mut.get_mut::<SignalHandles>() {
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
            entity.insert(SignalHandles::default());
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
