#![deny(missing_docs)]
//! # jonmo - Declarative Signals for Bevy
//!
//! jonmo provides a way to define reactive signal chains in Bevy using a declarative
//! builder pattern. Signals originate from sources (like component changes, resource changes,
//! or specific entities) and can be transformed (`map`), combined (`combine_with`), or
//! deduplicated (`dedupe`).
//!
//! The core building block is the [`Signal`] trait, representing a value that changes over time.
//! Chains are constructed starting with methods like [`SignalBuilder::from_component`] or
//! [`SignalBuilder::from_resource`], followed by combinators like [`SignalExt::map`] or
//! [`SignalExt::combine_with`]. Signal chains must implement `Clone` to be used with combinators
//! like `combine_with` or to be cloned into closures.
//!
//! Finally, a signal chain is activated by calling [`SignalExt::register`], which registers
//! the necessary Bevy systems and returns a [`SignalHandle`] for potential cleanup.
//! Cleaning up a handle removes *all* systems created by that specific `register` call
//! by decrementing reference counts. If systems were shared with other signal chains, cleaning up
//! one handle will only remove those shared systems if their reference count reaches zero.
//!
//! ## Execution Model
//!
//! Internally, jonmo builds and maintains a dependency graph of Bevy systems. Each frame,
//! the [`JonmoPlugin`] triggers the execution of this graph starting from the root systems
//! (created via `SignalBuilder::from_*` methods). It pipes the output (`Some(O)`) of a parent
//! system as the input (`In<O>`) to its children using type-erased runners. This traversal
//! continues down each branch until a system returns `None` (often represented by the
//! [`TERMINATE`] constant), which halts propagation along that specific path for the current frame.
//!
//! The signal propagation is managed internally by the [`JonmoPlugin`] which should be added
//! to your Bevy `App`.
//!
//! ## Example
//!
//! ```no_run
//! use bevy::prelude::*;
//! use jonmo::prelude::*; // Import prelude for common items
//!
//! #[derive(Component, Reflect, Clone, Default, PartialEq)]
//! #[reflect(Component)]
//! struct Value(i32);
//!
//! #[derive(Resource)]
//! struct UiEntities { main: Entity, text: Entity }
//!
//! fn setup_ui_declarative(world: &mut World) {
//!     // Assume ui_entities resource is populated
//!     let ui_entities = world.get_resource::<UiEntities>().unwrap();
//!     let entity = ui_entities.main;
//!     let text = ui_entities.text;
//!
//!     let text_node = text.clone(); // Clone for closure
//!     let signal_chain = SignalBuilder::from_component::<Value>(entity) // Start from Value component changes
//!         .map(dedupe) // Only propagate if the value is different from the last
//!         .map(move |In(value): In<Value>, mut cmd: Commands| { // Update text when value changes
//!             println!("Updating text with Value: {}", value.0);
//!             cmd.entity(text_node).insert(Text::from_section(value.0.to_string(), Default::default())); // Assuming Text setup
//!             TERMINATE // Signal ends here for this frame's execution path.
//!         });
//!
//!     // Register the systems and get a handle
//!     let handle = signal_chain.register(world);
//!     // Store handle if cleanup is needed later
//!     // handle.cleanup(world); // Example cleanup call
//! }
//!
//! fn main() {
//!     let mut app = App::new();
//!     app.add_plugins(DefaultPlugins)
//!         .add_plugins(JonmoPlugin) // Add the Jonmo plugin struct
//!         .register_type::<Value>()
//!         // ... setup entities and resources ...
//!         .add_systems(Startup, setup_ui_declarative);
//!     app.run();
//! }
//! ```

use bevy_app::prelude::*;
use bevy_ecs::{prelude::*, system::SystemId};
use bevy_log::prelude::*;
use bevy_reflect::{GetTypeRegistration, Typed, prelude::*};
use std::{
    collections::{HashMap, HashSet},
    marker::PhantomData,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

// --- Type Aliases for Boxed Registration Functions ---

/// Type alias for the boxed source registration function.
type SourceRegisterFn<O> =
    dyn Fn(&mut World) -> SystemId<In<()>, Option<O>> + Send + Sync + 'static;

/// Type alias for the boxed combine registration function.
type CombineRegisterFn<O1, O2> = dyn Fn(
        &mut World,
        UntypedSystemId,
        UntypedSystemId,
    ) -> (
        SystemId<In<(Option<O1>, Option<O2>)>, Option<(O1, O2)>>,
        Vec<UntypedSystemId>,
    ) + Send
    + Sync
    + 'static;

/// Type alias for the boxed map registration function.
type MapRegisterFn = dyn Fn(
        &mut World,
        UntypedSystemId, // Previous system's entity ID
    ) -> UntypedSystemId // Registered map system's entity ID
    + Send
    + Sync
    + 'static;

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
type UntypedSystemId = Entity;

/// Component storing metadata for signal system nodes, primarily for reference counting.
#[derive(Component)]
pub(crate) struct SignalNodeMetadata {
    /// Number of SignalHandles referencing this system node.
    ref_count: AtomicUsize,
}

impl SignalNodeMetadata {
    /// Creates metadata with an initial reference count of 1.
    fn new() -> Self {
        Self {
            ref_count: AtomicUsize::new(1),
        }
    }

    /// Atomically increments the reference count.
    fn increment(&self) {
        self.ref_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Atomically decrements the reference count and returns the *previous* count.
    fn decrement(&self) -> usize {
        self.ref_count.fetch_sub(1, Ordering::Release)
    }
}

/// Helper to register a system, add the [`SystemRunner`] component, and manage [`SignalNodeMetadata`].
///
/// Ensures the system is registered, attaches a runner component, and handles the
/// reference counting via `SignalNodeMetadata`. Returns the `SystemId` and indicates
/// if the node was newly created.
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

/// Internal trait handling the registration logic for different signal node types.
/// **Note:** This trait is intended for internal use only.
pub trait SignalBuilderInternal: Send + Sync + 'static {
    /// The logical output type of this signal node.
    type Item: Send + Sync + 'static;

    /// Registers the systems associated with this node and its predecessors in the `World`.
    /// Returns a `Vec<UntypedSystemId>` containing the entities of *all* systems
    /// registered or reference-counted during this specific registration call instance.
    fn register(&self, world: &mut World) -> Vec<UntypedSystemId>;
}

/// Represents a value that changes over time.
///
/// Signals are the core building block for reactive data flow. They are typically
/// created using methods on the [`SignalBuilder`] struct (e.g., [`SignalBuilder::from_component`])
/// and then transformed or combined using methods from the [`SignalExt`] trait.
pub trait Signal: Send + Sync + 'static {
    /// The type of value produced by this signal.
    type Item: Send + Sync + 'static;
}

/// Struct representing a source node in the signal chain definition. Implements [`Signal`].
#[derive(Clone)]
pub struct Source<O>
where
    O: Send + Sync + 'static,
{
    /// The type-erased function responsible for registering the source system.
    pub(crate) register_fn: Arc<SourceRegisterFn<O>>,
    _marker: PhantomData<O>,
}

// Implement internal registration logic
impl<O> SignalBuilderInternal for Source<O>
where
    O: Send + Sync + 'static,
{
    type Item = O;

    fn register(&self, world: &mut World) -> Vec<UntypedSystemId> {
        let system_id = (self.register_fn)(world);
        vec![system_id.entity()]
    }
}

// Implement the public Signal trait
impl<O> Signal for Source<O>
where
    O: Send + Sync + 'static,
{
    type Item = O;
}

/// Struct representing a map node in the signal chain definition. Implements [`Signal`].
/// Generic only over the previous signal (`Prev`) and the output type (`U`).
pub struct Map<Prev, U>
where
    Prev: SignalBuilderInternal,
    U: Send + Sync + 'static,
    <Prev as SignalBuilderInternal>::Item: Send + Sync + 'static,
{
    pub(crate) prev_signal: Prev,
    pub(crate) register_fn: Arc<MapRegisterFn>,
    _marker: PhantomData<U>,
}

// Add Clone implementation for Map
impl<Prev, U> Clone for Map<Prev, U>
where
    Prev: SignalBuilderInternal + Clone,
    U: Send + Sync + 'static,
    <Prev as SignalBuilderInternal>::Item: Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        Self {
            prev_signal: self.prev_signal.clone(),
            register_fn: self.register_fn.clone(),
            _marker: PhantomData,
        }
    }
}

// Implement internal registration logic for Map<Prev, U>
impl<Prev, U> SignalBuilderInternal for Map<Prev, U>
where
    Prev: SignalBuilderInternal,
    U: FromReflect + Send + Sync + 'static,
    <Prev as SignalBuilderInternal>::Item: FromReflect + Send + Sync + 'static,
{
    type Item = U;

    fn register(&self, world: &mut World) -> Vec<UntypedSystemId> {
        let mut prev_ids = self.prev_signal.register(world);
        if let Some(&prev_last_id_entity) = prev_ids.last() {
            let new_system_entity = (self.register_fn)(world, prev_last_id_entity);
            prev_ids.push(new_system_entity);
        } else {
            error!("Map signal parent registration returned empty ID list.");
        }
        prev_ids
    }
}

// Implement the public Signal trait for Map<Prev, U>
impl<Prev, U> Signal for Map<Prev, U>
where
    Prev: SignalBuilderInternal,
    U: Send + Sync + 'static,
    <Prev as SignalBuilderInternal>::Item: Send + Sync + 'static,
    Prev: Signal<Item = <Prev as SignalBuilderInternal>::Item>,
{
    type Item = U;
}

/// Struct representing a combine node in the signal chain definition. Implements [`Signal`].
pub struct Combine<Left, Right>
where
    Left: Signal + SignalBuilderInternal,
    Right: Signal + SignalBuilderInternal,
    <Left as Signal>::Item: Send + Sync + 'static,
    <Right as Signal>::Item: Send + Sync + 'static,
{
    pub(crate) left_signal: Left,
    pub(crate) right_signal: Right,
    pub(crate) register_fn: Arc<CombineRegisterFn<<Left as Signal>::Item, <Right as Signal>::Item>>,
    _marker: PhantomData<(<Left as Signal>::Item, <Right as Signal>::Item)>,
}

// Add Clone implementation for Combine
impl<Left, Right> Clone for Combine<Left, Right>
where
    Left: Signal + SignalBuilderInternal + Clone,
    Right: Signal + SignalBuilderInternal + Clone,
    <Left as Signal>::Item: Send + Sync + 'static,
    <Right as Signal>::Item: Send + Sync + 'static,
    Arc<CombineRegisterFn<<Left as Signal>::Item, <Right as Signal>::Item>>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            left_signal: self.left_signal.clone(),
            right_signal: self.right_signal.clone(),
            register_fn: self.register_fn.clone(),
            _marker: PhantomData,
        }
    }
}

// Implement internal registration logic
impl<Left, Right> SignalBuilderInternal for Combine<Left, Right>
where
    Left: Signal + SignalBuilderInternal,
    Right: Signal + SignalBuilderInternal,
    <Left as Signal>::Item: Send + Sync + 'static,
    <Right as Signal>::Item: Send + Sync + 'static,
{
    type Item = (<Left as Signal>::Item, <Right as Signal>::Item);

    fn register(&self, world: &mut World) -> Vec<UntypedSystemId> {
        let mut left_ids = self.left_signal.register(world);
        let mut right_ids = self.right_signal.register(world);

        let combined_ids = if let (Some(&left_last_id), Some(&right_last_id)) =
            (left_ids.last(), right_ids.last())
        {
            let (_combine_system_id, combine_node_ids) =
                (self.register_fn)(world, left_last_id, right_last_id);
            combine_node_ids
        } else {
            error!("CombineSignal parent registration returned empty ID list(s).");
            Vec::new()
        };

        left_ids.append(&mut right_ids);
        left_ids.extend(combined_ids);
        left_ids
    }
}

// Implement the public Signal trait
impl<Left, Right> Signal for Combine<Left, Right>
where
    Left: Signal + SignalBuilderInternal,
    Right: Signal + SignalBuilderInternal,
    <Left as Signal>::Item: Send + Sync + 'static,
    <Right as Signal>::Item: Send + Sync + 'static,
{
    type Item = (<Left as Signal>::Item, <Right as Signal>::Item);
}

/// Handle returned by [`SignalExt::register`] used for cleaning up the registered signal chain.
///
/// Contains the list of all system entities created by the specific `register` call
/// that produced this handle. Dropping the handle does *not* automatically clean up.
/// Use the [`cleanup`](SignalHandle::cleanup) method for explicit cleanup.
#[derive(Clone, Debug)]
pub struct SignalHandle(Vec<UntypedSystemId>);

impl SignalHandle {
    /// Decrements the reference count for each system associated with this handle.
    /// If a system's reference count reaches zero, it removes the system's node
    /// from the internal signal graph and despawns its associated entity from the Bevy `World`.
    pub fn cleanup(self, world: &mut World) {
        if let Some(mut propagator) = world.remove_resource::<SignalPropagator>() {
            let mut nodes_to_remove = Vec::new();
            for system_entity in &self.0 {
                if let Some(metadata) = world.get::<SignalNodeMetadata>(*system_entity) {
                    if metadata.decrement() == 1 {
                        nodes_to_remove.push(*system_entity);
                    }
                } else {
                    warn!(
                        "SignalNodeMetadata not found for system {:?} during cleanup.",
                        system_entity
                    );
                }
            }

            for entity_to_remove in nodes_to_remove {
                propagator.remove_graph_node(world, entity_to_remove);
            }

            world.insert_resource(propagator);
        } else {
            warn!(
                "SignalPropagator not found during cleanup. Cannot decrement reference counts or remove nodes."
            );
            for system_entity in self.0 {
                if let Ok(entity_mut) = world.get_entity_mut(system_entity) {
                    entity_mut.despawn();
                }
            }
        }
    }
}

/// Helper to create a source signal node. Wraps the registration function
/// in the `Source` struct, boxing the function.
pub(crate) fn create_source<F, O>(register_fn: F) -> Source<O>
where
    O: Send + Sync + 'static,
    F: Fn(&mut World) -> SystemId<In<()>, Option<O>> + Send + Sync + 'static,
{
    Source {
        register_fn: Arc::new(register_fn),
        _marker: std::marker::PhantomData,
    }
}

/// Provides static methods for creating new signal chains (source signals).
/// Use methods like [`SignalBuilder::from_component`] or [`SignalBuilder::from_system`]
/// to start building a signal chain.
pub struct SignalBuilder;

// Static methods to start signal chains, now associated with SignalBuilder struct
impl SignalBuilder {
    /// Creates a signal chain starting from a custom Bevy system.
    ///
    /// The provided system should take `In<()>` and return `Option<O>`.
    /// This system will be registered as a root node for signal propagation.
    /// The system `F` must be `Clone` as it's captured for registration.
    pub fn from_system<O, M, F>(system: F) -> Source<O>
    where
        O: FromReflect + Send + Sync + 'static,
        F: IntoSystem<In<()>, Option<O>, M> + Send + Sync + Clone + 'static,
        M: Send + Sync + 'static,
    {
        let register_fn = move |world: &mut World| {
            let system_id = register_signal::<(), O, M>(world, system.clone());
            mark_signal_root(world, system_id.entity());
            system_id
        };
        create_source(register_fn)
    }

    /// Creates a signal chain starting from a specific entity.
    ///
    /// The signal will emit the `Entity` ID whenever the propagation starts from this source.
    /// Useful for chains that operate on or react to changes related to this entity.
    /// Internally uses [`SignalBuilder::from_system`] with [`entity_root`].
    pub fn from_entity(entity: Entity) -> Source<Entity> {
        Self::from_system(entity_root(entity))
    }

    /// Creates a signal chain that starts by observing changes to a specific component `C`
    /// on a given `entity`.
    ///
    /// The signal emits the new value of the component `C` whenever it changes on the entity.
    /// Requires the component `C` to implement `Component`, `FromReflect`, `Clone`, `Send`, `Sync`, and `'static`.
    /// Internally uses [`SignalBuilder::from_system`].
    pub fn from_component<C>(entity: Entity) -> Source<C>
    where
        C: Component + FromReflect + Clone + Send + Sync + 'static,
    {
        let component_query_system =
            move |_: In<()>, query: Query<&'static C, Changed<C>>| query.get(entity).ok().cloned();
        Self::from_system(component_query_system)
    }

    /// Creates a signal chain that starts by observing changes to a specific resource `R`.
    ///
    /// The signal emits the new value of the resource `R` whenever it changes.
    /// Requires the resource `R` to implement `Resource`, `FromReflect`, `Clone`, `Send`, `Sync`, and `'static`.
    /// Internally uses [`SignalBuilder::from_system`].
    pub fn from_resource<R>() -> Source<R>
    where
        R: Resource + FromReflect + Clone + Send + Sync + 'static,
    {
        let resource_query_system = move |_: In<()>, res: Res<R>| {
            if res.is_changed() {
                Some(res.clone())
            } else {
                None
            }
        };
        Self::from_system(resource_query_system)
    }
}

/// Extension trait providing combinator methods for types implementing [`Signal`],
/// [`SignalBuilderInternal`], and [`Clone`].
pub trait SignalExt: Signal + SignalBuilderInternal + Clone {
    /// Appends a transformation step to the signal chain using a Bevy system.
    ///
    /// The provided `system` takes the output `Item` of the previous step (wrapped in `In<Item>`)
    /// and returns an `Option<U>`. If it returns `Some(U)`, `U` is propagated to the next step.
    /// If it returns `None` (or [`TERMINATE`]), propagation along this branch stops for the frame.
    ///
    /// The system `F` must be `Clone` as it's captured for registration.
    /// Returns a [`Map`] signal node.
    fn map<U, M, F>(self, system: F) -> Map<Self, U>
    where
        Self: Sized,
        <Self as SignalBuilderInternal>::Item: Send + Sync + 'static,
        <Self as SignalBuilderInternal>::Item: FromReflect + Send + Sync + 'static,
        U: FromReflect + Send + Sync + 'static,
        F: IntoSystem<In<<Self as SignalBuilderInternal>::Item>, Option<U>, M>
            + Send
            + Sync
            + Clone
            + 'static,
        M: Send + Sync + 'static;

    /// Combines this signal with another signal (`other`), producing a new signal that emits
    /// a tuple `(Self::Item, S2::Item)` of the outputs of both signals.
    ///
    /// The new signal emits a value only when *both* input signals have emitted at least one
    /// value since the last combined emission. It caches the latest value from each input.
    /// Both `self` and `other` must implement `Clone`.
    /// Returns a [`Combine`] signal node.
    fn combine_with<S2>(self, other: S2) -> Combine<Self, S2>
    where
        Self: Sized,
        S2: Signal + SignalBuilderInternal + Clone,
        <Self as Signal>::Item: FromReflect
            + GetTypeRegistration
            + Typed
            + Send
            + Sync
            + Clone
            + 'static
            + std::fmt::Debug,
        <S2 as Signal>::Item: FromReflect
            + GetTypeRegistration
            + Typed
            + Send
            + Sync
            + Clone
            + 'static
            + std::fmt::Debug;

    /// Registers all the systems defined in this signal chain into the Bevy `World`.
    ///
    /// This activates the signal chain. It traverses the internal representation (calling
    /// [`SignalBuilderInternal::register`] recursively), registers each required Bevy system
    /// (or increments its reference count if already registered), connects them in the
    /// [`SignalPropagator`], and marks the source system(s) as roots.
    ///
    /// Returns a [`SignalHandle`] which can be used later to [`cleanup`](SignalHandle::cleanup)
    /// the systems created or referenced specifically by *this* `register` call.
    fn register(&self, world: &mut World) -> SignalHandle;
}

// Implement SignalExt for any type T that implements Signal + SignalBuilderInternal + Clone
impl<T> SignalExt for T
where
    T: Signal + SignalBuilderInternal<Item = <T as Signal>::Item> + Clone,
{
    fn map<U, M, F>(self, system: F) -> Map<Self, U>
    where
        <T as SignalBuilderInternal>::Item: Send + Sync + 'static,
        <T as SignalBuilderInternal>::Item: FromReflect + Send + Sync + 'static,
        U: FromReflect + Send + Sync + 'static,
        F: IntoSystem<In<<T as SignalBuilderInternal>::Item>, Option<U>, M>
            + Send
            + Sync
            + Clone
            + 'static,
        M: Send + Sync + 'static,
    {
        let system_clone = system.clone();

        let register_fn = Arc::new(
            move |world: &mut World, prev_last_id_entity: UntypedSystemId| -> UntypedSystemId {
                let system_id = register_signal::<<T as SignalBuilderInternal>::Item, U, M>(
                    world,
                    system_clone.clone(),
                );

                let system_entity = system_id.entity();

                pipe_signal(world, prev_last_id_entity, system_entity);

                system_entity
            },
        );

        Map {
            prev_signal: self,
            register_fn,
            _marker: PhantomData,
        }
    }
    fn combine_with<S2>(self, other: S2) -> Combine<Self, S2>
    where
        S2: Signal + SignalBuilderInternal + Clone,
        <Self as Signal>::Item: FromReflect
            + GetTypeRegistration
            + Typed
            + Send
            + Sync
            + Clone
            + 'static
            + std::fmt::Debug,
        <S2 as Signal>::Item: FromReflect
            + GetTypeRegistration
            + Typed
            + Send
            + Sync
            + Clone
            + 'static
            + std::fmt::Debug,
    {
        let register_fn = move |world: &mut World,
                                left_id_entity: UntypedSystemId,
                                right_id_entity: UntypedSystemId| {
            combine_signal::<<Self as Signal>::Item, <S2 as Signal>::Item>(
                world,
                left_id_entity,
                right_id_entity,
            )
        };

        Combine {
            left_signal: self,
            right_signal: other,
            register_fn: Arc::new(register_fn)
                as Arc<CombineRegisterFn<<Self as Signal>::Item, <S2 as Signal>::Item>>,
            _marker: PhantomData,
        }
    }

    fn register(&self, world: &mut World) -> SignalHandle {
        let all_system_ids = <T as SignalBuilderInternal>::register(self, world);
        SignalHandle(all_system_ids)
    }
}

/// System that drives signal propagation by calling [`SignalPropagator::execute`].
/// Added to the `Update` schedule by the [`JonmoPlugin`]. This system runs once per frame.
/// It temporarily removes the [`SignalPropagator`] resource to allow mutable access to the `World`
/// during system execution within the propagator.
pub(crate) fn process_signals(world: &mut World) {
    if let Some(propagator) = world.remove_resource::<SignalPropagator>() {
        propagator.execute(world);
        world.insert_resource(propagator);
    }
}

/// The Bevy plugin required for `jonmo` signals to function.
///
/// Adds the necessary [`SignalPropagator`] resource and the system that drives
/// signal propagation ([`process_signals`]) to the `Update` schedule.
///
/// ```no_run
/// use bevy::prelude::*;
/// use jonmo::prelude::*; // Use prelude
///
/// App::new()
///     .add_plugins(DefaultPlugins)
///     .add_plugins(JonmoPlugin) // Add the plugin here
///     // ... other app setup ...
///     .run();
/// ```
#[derive(Default)]
pub struct JonmoPlugin;

impl Plugin for JonmoPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, process_signals)
            .init_resource::<SignalPropagator>();
    }
}

/// Commonly used items for working with `jonmo` signals.
///
/// This prelude includes the core traits, structs, and functions needed to
/// define and manage signal chains. It excludes internal implementation details
/// like [`SignalBuilderInternal`].
///
/// ```
/// use jonmo::prelude::*;
/// ```
pub mod prelude {
    pub use crate::{
        Combine, JonmoPlugin, Map, Signal, SignalBuilder, SignalExt, SignalHandle, Source,
        TERMINATE, dedupe, entity_root,
    };
    // Note: SignalBuilderInternal is intentionally excluded
}
