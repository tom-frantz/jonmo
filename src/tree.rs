use bevy_ecs::{prelude::*, system::SystemId};
use bevy_log::prelude::*;
use bevy_reflect::{FromReflect, GetTypeRegistration, PartialReflect, Typed}; // Removed unused Reflect
use std::{
    collections::{HashMap, HashSet},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

/// Creates a simple signal source system that always outputs the given `entity`.
/// Useful as a starting point for signals related to a specific entity.
/// The returned system takes `In<()>` and returns `Option<Entity>`.
///
/// ```
/// use bevy::prelude::*;
/// use jonmo::entity_root;
///
/// let mut world = World::new();
/// let entity = world.spawn_empty().id();
/// let root_fn = entity_root(entity);
///
/// // Create a system from the function
/// let mut system = IntoSystem::into_system(root_fn);
/// system.initialize(&mut world);
///
/// // Run the system
/// let result = system.run((), &mut world);
/// assert_eq!(result, Some(entity));
/// ```
pub fn entity_root(entity: Entity) -> impl Fn(In<()>) -> Option<Entity> + Clone {
    move |_: In<()>| Some(entity)
}

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

// Use Entity as an untyped SystemId for internal bookkeeping
pub(crate) type UntypedSystemId = Entity;

/// Component storing metadata for signal system nodes, primarily for reference counting.
#[derive(Component)]
pub(crate) struct SignalNodeMetadata {
    /// Number of SignalHandles referencing this system node.
    pub(crate) ref_count: AtomicUsize,
}

impl SignalNodeMetadata {
    /// Creates metadata with an initial reference count of 1.
    pub(crate) fn new() -> Self {
        Self {
            ref_count: AtomicUsize::new(1),
        }
    }

    /// Atomically increments the reference count.
    pub(crate) fn increment(&self) {
        self.ref_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Atomically decrements the reference count and returns the *previous* count.
    pub(crate) fn decrement(&self) -> usize {
        self.ref_count.fetch_sub(1, Ordering::Release)
    }
}

/// Helper to register a system, add the [`SystemRunner`] component, and manage [`SignalNodeMetadata`].
///
/// Ensures the system is registered, attaches a runner component, and handles the
/// reference counting via `SignalNodeMetadata`. Returns the `SystemId`.
pub fn register_signal<I, O, M>(
    world: &mut World,
    system: impl IntoSystem<In<I>, Option<O>, M> + 'static,
) -> SystemId<In<I>, Option<O>>
where
    I: FromReflect + Send + Sync + 'static,
    O: Send + FromReflect + Send + Sync + 'static,
    M: Send + Sync + 'static,
{
    let system_id = world.register_system(system);
    let system_entity = system_id.entity();
    let mut entity_mut = world.entity_mut(system_entity);

    if entity_mut.get::<SystemRunner>().is_none() {
        entity_mut.insert(SystemRunner {
            run: Arc::new(Box::new(move |world, input| {
                let system_id = SystemId::<In<I>, Option<O>>::from_entity(system_entity);
                match I::from_reflect(input.as_ref()) {
                    Some(typed_input) => world
                        .run_system_with_input(system_id, typed_input)
                        .ok()
                        .flatten()
                        .map(|val| Box::new(val) as Box<dyn PartialReflect>),
                    None => {
                        warn!(
                            "failed to downcast input for system {:?}: {:?}",
                            system_entity,
                            input.reflect_type_path()
                        );
                        None
                    }
                }
            })),
        });
    }

    if let Some(metadata) = entity_mut.get::<SignalNodeMetadata>() {
        metadata.increment();
    } else {
        entity_mut.insert(SignalNodeMetadata::new());
    }

    system_id
}

/// Helper to mark a system entity as a root in the [`SignalPropagator`].
///
/// Root systems are the starting points for signal propagation during the `Update` phase.
pub fn mark_signal_root(world: &mut World, system_entity: UntypedSystemId) {
    if let Some(mut propagator) = world.get_resource_mut::<SignalPropagator>() {
        propagator.add_root(system_entity);
    } else {
        warn!(
            "SignalPropagator resource not found while registering root {:?}",
            system_entity
        );
    }
}

/// Helper to connect two system entities in the [`SignalPropagator`] graph.
///
/// Establishes a parent-child relationship, indicating that the output of the `source`
/// system should be passed as input to the `target` system during propagation.
pub fn pipe_signal(world: &mut World, source: UntypedSystemId, target: UntypedSystemId) {
    if let Some(mut propagator) = world.get_resource_mut::<SignalPropagator>() {
        propagator.add_child(source, target);
    } else {
        warn!(
            "SignalPropagator resource not found while piping {:?} -> {:?}",
            source, target
        );
    }
}

/// Helper to register the systems needed for combining two signal branches.
///
/// Creates wrapper systems and a final combining system, managing reference counts
/// and connecting them in the [`SignalPropagator`]. Returns the `SystemId` of the
/// final combiner system and a Vec of all system entities involved in this combine step.
pub fn combine_signal<OLeft, ORight>(
    world: &mut World,
    left_last_entity: UntypedSystemId,
    right_last_entity: UntypedSystemId,
) -> (
    SystemId<In<(Option<OLeft>, Option<ORight>)>, Option<(OLeft, ORight)>>,
    Vec<UntypedSystemId>,
)
where
    OLeft: FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static,
    ORight: FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static,
{
    let left_wrapper_id = register_signal::<OLeft, (Option<OLeft>, Option<ORight>), _>(
        world,
        |In(left_val): In<OLeft>| Some((Some(left_val), None::<ORight>)),
    );
    pipe_signal(world, left_last_entity, left_wrapper_id.entity());

    let right_wrapper_id = register_signal::<ORight, (Option<OLeft>, Option<ORight>), _>(
        world,
        |In(right_val): In<ORight>| Some((None::<OLeft>, Some(right_val))),
    );
    pipe_signal(world, right_last_entity, right_wrapper_id.entity());

    let combine_id = register_signal::<_, _, _>(
        world,
        move |In((left_opt, right_opt)): In<(Option<OLeft>, Option<ORight>)>,
              mut left_cache: Local<Option<OLeft>>,
              mut right_cache: Local<Option<ORight>>| {
            if left_opt.is_some() {
                *left_cache = left_opt;
            }
            if right_opt.is_some() {
                *right_cache = right_opt;
            }
            if left_cache.is_some() && right_cache.is_some() {
                left_cache.take().zip(right_cache.take())
            } else {
                None
            }
        },
    );

    pipe_signal(world, left_wrapper_id.entity(), combine_id.entity());
    pipe_signal(world, right_wrapper_id.entity(), combine_id.entity());

    let registered_entities = vec![
        left_wrapper_id.entity(),
        right_wrapper_id.entity(),
        combine_id.entity(),
    ];

    (combine_id, registered_entities)
}

/// Component holding the type-erased system runner function.
///
/// This component is attached to the entity associated with each registered signal system.
/// It contains an `Arc<Box<dyn Fn(...)>>` that captures the specific `SystemId` and
/// handles the type-erased execution logic, including downcasting inputs and boxing outputs.
#[derive(Component, Clone)]
pub(crate) struct SystemRunner {
    /// The type-erased function to execute the system.
    pub(crate) run: Arc<
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
        (self.run)(world, input)
    }
}

/// Resource managing the signal graph and propagation logic.
///
/// Stores the relationships between signal systems (nodes) and identifies the root nodes
/// to start propagation from. The `execute` method traverses the graph depth-first each frame.
#[derive(Resource, Default)]
pub(crate) struct SignalPropagator {
    /// Stores the graph structure. Maps a system entity (parent) to an optional set
    /// of its children system entities. `None` indicates a node with no children yet.
    nodes: HashMap<Entity, Option<HashSet<Entity>>>,
    /// Set of system entities that are roots of signal chains. Propagation starts from these.
    roots: HashSet<Entity>,
}

impl SignalPropagator {
    /// Adds a system entity as a root node, ensuring it's also present in the `nodes` map.
    /// Avoids adding duplicates to the `roots` set.
    pub(crate) fn add_root(&mut self, entity: Entity) -> Entity {
        self.add_node(entity);
        if !self.roots.contains(&entity) {
            self.roots.insert(entity);
        }
        entity
    }

    /// Adds a child system entity to a parent system entity in the graph.
    /// Ensures both nodes exist in the `nodes` map and initializes the child set if needed.
    /// Avoids adding duplicate children.
    pub(crate) fn add_child(&mut self, parent_entity: Entity, child_entity: Entity) -> Entity {
        self.add_node(parent_entity);
        self.add_node(child_entity);

        let children = self.nodes.entry(parent_entity).or_insert(None);
        if children.is_none() {
            *children = Some(HashSet::new());
        }

        if let Some(set) = children {
            set.insert(child_entity);
        }

        child_entity
    }

    /// Ensures a node exists in the `nodes` map, initializing its children to `None` if new.
    fn add_node(&mut self, entity: Entity) -> Entity {
        self.nodes.entry(entity).or_insert(None);
        entity
    }

    /// Removes a node from the graph structure and despawns its entity.
    /// Called when a node's reference count reaches zero.
    pub(crate) fn remove_graph_node(&mut self, world: &mut World, entity: Entity) {
        if !self.nodes.contains_key(&entity) {
            if let Ok(entity_mut) = world.get_entity_mut(entity) {
                warn!(
                    "Despawning entity {:?} found in world but not in propagator nodes map during cleanup.",
                    entity
                );
                entity_mut.despawn();
            }
            return;
        }

        debug!("Removing graph node {:?}", entity);

        self.roots.remove(&entity);

        let parents: Vec<Entity> = self
            .nodes
            .iter()
            .filter_map(|(parent, children_opt)| {
                if let Some(children) = children_opt {
                    if children.contains(&entity) {
                        Some(*parent)
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();

        for parent in parents {
            if let Some(Some(children)) = self.nodes.get_mut(&parent) {
                children.remove(&entity);
                debug!("Removed child {:?} from parent {:?}", entity, parent);
            }
        }

        self.nodes.remove(&entity);
        debug!("Removed node {:?} from nodes map.", entity);

        if let Ok(entity_commands) = world.get_entity_mut(entity) {
            debug!("Despawning signal system entity {:?}", entity);
            entity_commands.despawn();
        } else {
            warn!(
                "Attempted to despawn entity {:?} during signal cleanup, but it was already gone.",
                entity
            );
        }
    }

    /// Executes all registered signal chains starting from the root nodes for the current frame.
    ///
    /// Iterates through the `roots`, runs their corresponding systems via [`SystemRunner`],
    /// and if a root system produces `Some(output)`, recursively calls `process_children`
    /// to propagate the output down the graph. This happens once per frame via the `process_signals` system.
    pub(crate) fn execute(&self, world: &mut World) {
        for &root_entity in &self.roots {
            if let Some(runner) = world
                .get_entity(root_entity)
                .ok()
                .and_then(|e| e.get::<SystemRunner>())
                .cloned()
            {
                if let Some(output) = runner.run(world, Box::new(())) {
                    self.process_children(root_entity, world, output);
                }
            } else {
                warn!(
                    "SystemRunner component not found for root entity {:?}",
                    root_entity
                );
            }
        }
    }

    /// Recursively processes the children of a given parent node during the current frame's propagation.
    ///
    /// For each child, it retrieves its [`SystemRunner`], executes it with the `input`
    /// from the parent. If the child system produces `Some(output)`, it recursively calls itself
    /// for the child's children, continuing the propagation. If the child system returns `None`,
    /// the propagation stops along this branch for this frame.
    fn process_children(
        &self,
        parent_entity: Entity,
        world: &mut World,
        input: Box<dyn PartialReflect>,
    ) {
        if let Some(Some(children)) = self.nodes.get(&parent_entity) {
            for &child_entity in children {
                if let Some(runner) = world
                    .get_entity(child_entity)
                    .ok()
                    .and_then(|e| e.get::<SystemRunner>())
                    .cloned()
                {
                    if let Some(output) = runner.run(world, input.clone_value()) {
                        self.process_children(child_entity, world, output);
                    }
                } else {
                    warn!(
                        "SystemRunner component not found for child entity {:?}",
                        child_entity
                    );
                }
            }
        }
    }
}
