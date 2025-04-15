#![deny(missing_docs)]
//! # jonmo - Declarative Signals for Bevy
//!
//! jonmo provides a way to define reactive signal chains in Bevy using a declarative
//! builder pattern. Signals originate from sources (like component changes, resource changes,
//! or specific entities) and can be transformed (`map`), combined (`combine_with`), or
//! deduplicated (`dedupe`).
//!
//! The core building block is the [`Signal`] struct, which represents a potential flow of data.
//! Chains are constructed starting with methods like [`Signal::from_component`] or
//! [`Signal::from_resource`], followed by combinators like [`Signal::map`] or
//! [`Signal::combine_with`].
//!
//! Finally, a signal chain is activated by calling [`Signal::register`], which registers
//! the necessary Bevy systems and returns a [`SignalHandle`] for potential cleanup.
//!
//! ## Execution Model
//!
//! Internally, jonmo builds and maintains a dependency graph of Bevy systems. Each frame,
//! the [`JonmoPlugin`] traverses this graph starting from the root systems (created via
//! `Signal::from_*` methods or marked explicitly with `mark_signal_root`). It pipes the
//! output (`Some(O)`) of a parent system as the input (`In<O>`) to its children. This
//! traversal continues down each branch until a system returns `None` (often represented
//! by the [`TERMINATE`] constant), which halts propagation along that specific path for
//! the current frame.
//!
//! The signal propagation is managed internally by the [`JonmoPlugin`] which should be added
//! to your Bevy `App`.
//!
//! ## Example
//!
//! ```no_run
//! use bevy::prelude::*;
//! use jonmo::*;
//!
//! #[derive(Component, Reflect, Clone, Default, PartialEq)]
//! #[reflect(Component)]
//! struct Value(i32);
//!
//! #[derive(Resource)]
//! struct UiEntities { main: Entity, text: Entity }
//!
//! fn setup_ui_declarative(world: &mut World) {
//!     let ui_entities = world.get_resource::<UiEntities>().unwrap();
//!     let entity = ui_entities.main;
//!     let text = ui_entities.text;
//!
//!     let text_node = text.clone(); // Clone for closure
//!     let signal_chain = Signal::from_component::<Value>(entity) // Start from Value component changes
//!         .map(dedupe) // Only propagate if the value is different from the last
//!         .map(move |In(value): In<Value>, mut cmd: Commands| { // Update text when value changes
//!             println!("Updating text with Value: {}", value.0);
//!             cmd.entity(text_node).insert(Text(value.0.to_string()));
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
    sync::Arc,
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

/// A system that can be used with [`Signal::map`] to prevent propagation
/// if the incoming value is the same as the previous one.
///
/// Requires the value type `T` to implement `PartialEq`, `Clone`, `Send`, `Sync`, and `'static`.
/// It uses [`Local`] state to store the previous value.
///
/// ```
/// use bevy::prelude::*;
/// use jonmo::dedupe;
///
/// #[derive(Clone, PartialEq, Debug)]
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
    T: PartialEq + Clone + Send + Sync + 'static + std::fmt::Debug,
{
    let mut changed = false;
    if let Some(ref p) = *cache {
        if *p != current {
            // Value changed compared to the previous one
            changed = true;
        }
        // else: Value is the same, changed remains false
    } else {
        // No previous value stored, so this is the first or changed
        changed = true;
    }

    if changed {
        // Update the stored previous value and propagate the current one
        *cache = Some(current.clone());
        Some(current)
    } else {
        // Value is the same as the previous one, terminate propagation
        None
    }
}

// Use Entity as an untyped SystemId for internal bookkeeping
type UntypedSystemId = Entity;

/// Helper to register a system and add the [`SystemRunner`] component.
///
/// Ensures that the system is registered with the world and attaches a runner
/// component that allows the [`SignalPropagator`] to execute the system with
/// type-erased inputs and outputs. It avoids adding the runner if it already exists.
pub fn register_signal<I, O, M>(
    world: &mut World,
    system: impl IntoSystem<In<I>, Option<O>, M> + 'static,
) -> SystemId<In<I>, Option<O>>
where
    I: FromReflect + Send + Sync + 'static,
    O: Send + FromReflect + Send + Sync + 'static,
{
    let system_id = world.register_system(system);
    let system_entity = system_id.entity();
    if world.get::<SystemRunner>(system_entity).is_none() {
        world.entity_mut(system_entity).insert(SystemRunner {
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
/// Creates two wrapper systems to adapt the outputs of the `left` and `right` branches
/// into a common `(Option<OLeft>, Option<ORight>)` type. It then registers a final
/// combining system that uses `Local` state to cache values from each branch and
/// emits an `(OLeft, ORight)` tuple only when a value has been received from both.
/// Connects the original branches to the wrappers, and the wrappers to the final combiner
/// in the [`SignalPropagator`].
pub fn combine_signal<OLeft, ORight>(
    world: &mut World,
    left: UntypedSystemId,
    right: UntypedSystemId,
) -> SystemId<In<(Option<OLeft>, Option<ORight>)>, Option<(OLeft, ORight)>>
where
    OLeft: FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static,
    ORight: FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static,
{
    let left_wrapper_id = register_signal::<OLeft, (Option<OLeft>, Option<ORight>), _>(
        world,
        |In(left_val): In<OLeft>| Some((Some(left_val), None::<ORight>)),
    );
    pipe_signal(world, left, left_wrapper_id.entity());

    let right_wrapper_id = register_signal::<ORight, (Option<OLeft>, Option<ORight>), _>(
        world,
        |In(right_val): In<ORight>| Some((None::<OLeft>, Some(right_val))),
    );
    pipe_signal(world, right, right_wrapper_id.entity());

    let combine_id = register_signal(
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

    combine_id
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
                + Sync,
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

    /// Recursively removes a node and its downstream connections from the graph.
    ///
    /// It despawns the corresponding entity from the `World`. If a child node has other
    /// parents, only the connection from the removed parent is severed; the child node
    /// itself is not removed or despawned unless this was its last parent.
    pub(crate) fn remove_node(&mut self, world: &mut World, entity: Entity) {
        if !self.nodes.contains_key(&entity) {
            return;
        }

        if let Some(Some(children)) = self.nodes.get(&entity).cloned() {
            for child in children {
                if self.nodes.contains_key(&child) {
                    let mut other_parents_exist = false;
                    for (parent, maybe_children) in &self.nodes {
                        if *parent != entity {
                            if let Some(child_set) = maybe_children {
                                if child_set.contains(&child) {
                                    other_parents_exist = true;
                                    break;
                                }
                            }
                        }
                    }
                    if !other_parents_exist {
                        self.remove_node(world, child);
                    }
                }
            }
        }

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
            }
        }

        self.roots.remove(&entity);
        self.nodes.remove(&entity);

        if let Ok(entity_commands) = world.get_entity_mut(entity) {
            info!("Despawning signal system entity {:?}", entity);
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
                // Run the root system (input is always () for roots)
                if let Some(output) = runner.run(world, Box::new(())) {
                    // If the root produced output, start propagating to its children
                    self.process_children(root_entity, world, output);
                }
                // If runner.run returned None, propagation stops here for this root this frame.
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
                    // Run the child system with the input from the parent
                    if let Some(output) = runner.run(world, input.clone_value()) {
                        // If the child produced output, continue propagation to its children
                        self.process_children(child_entity, world, output);
                    }
                    // If runner.run returned None, propagation stops here for this child this frame.
                } else {
                    warn!(
                        "SystemRunner component not found for child entity {:?}",
                        child_entity
                    );
                }
            }
        }
        // If node has no children (or children set is None), propagation naturally stops here.
    }
}

/// Trait abstracting over the different types of signal chain nodes
/// (source, map, combine) during the registration phase.
pub(crate) trait ISignal: Send + Sync {
    /// Registers the systems associated with this node and its predecessors in the `World`.
    /// Returns the `UntypedSystemId` (Entity) of the *last* system registered by this node.
    fn register(&self, world: &mut World) -> UntypedSystemId;
}

/// Struct representing a source node in the signal chain definition.
/// Contains the function that registers the initial system.
pub(crate) struct SourceSignal<F>
where
    F: Fn(&mut World) -> UntypedSystemId + Send + Sync + 'static + ?Sized,
{
    /// The function responsible for registering the source system.
    pub(crate) register_fn: Arc<F>,
}

impl<F> ISignal for SourceSignal<F>
where
    F: Fn(&mut World) -> UntypedSystemId + Send + Sync + 'static + ?Sized,
{
    /// Executes the stored `register_fn` to register the source system.
    fn register(&self, world: &mut World) -> UntypedSystemId {
        (self.register_fn)(world)
    }
}

/// Struct representing a map node in the signal chain definition.
/// Contains a reference to the previous node and the function to register the mapping system.
pub(crate) struct MapSignal<F>
where
    F: Fn(&mut World, UntypedSystemId) -> UntypedSystemId + Send + Sync + 'static + ?Sized,
{
    /// The preceding node in the signal chain definition.
    pub(crate) prev_signal: Arc<dyn ISignal>,
    /// The function responsible for registering the map system and piping it.
    pub(crate) register_fn: Arc<F>,
}

impl<F> ISignal for MapSignal<F>
where
    F: Fn(&mut World, UntypedSystemId) -> UntypedSystemId + Send + Sync + 'static + ?Sized,
{
    /// Recursively registers the previous node, then executes the stored `register_fn`
    /// to register the map system and connect it to the previous one.
    fn register(&self, world: &mut World) -> UntypedSystemId {
        let prev_id = self.prev_signal.register(world);
        (self.register_fn)(world, prev_id)
    }
}

/// Struct representing a combine node in the signal chain definition.
/// Contains references to the two parent nodes and the function to register the combining systems.
pub(crate) struct CombineSignal<F>
where
    F: Fn(&mut World, UntypedSystemId, UntypedSystemId) -> UntypedSystemId
        + Send
        + Sync
        + 'static
        + ?Sized,
{
    /// The left parent node in the signal chain definition.
    pub(crate) left_signal: Arc<dyn ISignal>,
    /// The right parent node in the signal chain definition.
    pub(crate) right_signal: Arc<dyn ISignal>,
    /// The function responsible for registering the combine systems and piping them.
    pub(crate) register_fn: Arc<F>,
}

impl<F> ISignal for CombineSignal<F>
where
    F: Fn(&mut World, UntypedSystemId, UntypedSystemId) -> UntypedSystemId
        + Send
        + Sync
        + 'static
        + ?Sized,
{
    /// Recursively registers both parent nodes, then executes the stored `register_fn`
    /// to register the necessary combining/wrapper systems and connect them.
    fn register(&self, world: &mut World) -> UntypedSystemId {
        let left_id = self.left_signal.register(world);
        let right_id = self.right_signal.register(world);
        (self.register_fn)(world, left_id, right_id)
    }
}

/// Handle returned by [`Signal::register`] used for cleaning up the registered signal chain.
///
/// Dropping the handle does *not* automatically clean up the signal. Cleanup must be
/// explicitly requested using the [`cleanup`](SignalHandle::cleanup) method.
// Add Debug derive for potential use in user code assertions
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SignalHandle(UntypedSystemId);

impl SignalHandle {
    /// Removes the system associated with this handle and all its downstream signal systems
    /// from the internal signal graph and despawns their associated entities from the Bevy `World`.
    ///
    /// This operation traverses the signal graph downstream from the handle's system.
    /// If a downstream system is only reachable through this path, it will be removed.
    /// If a downstream system has other parents in the signal graph, it will *not* be removed.
    /// Upstream systems are never affected by cleaning up a downstream handle.
    ///
    /// **Note:** Requires mutable access to the `World` to despawn entities and access the
    /// internal `SignalPropagator` resource.
    ///
    /// ```no_run
    /// use bevy::prelude::*;
    /// use jonmo::*;
    ///
    /// #[derive(Component, Reflect, Clone, Default, PartialEq)]
    /// #[reflect(Component)]
    /// struct MyData(i32);
    ///
    /// let mut world = World::new();
    /// world.init_resource::<SignalPropagator>(); // Need the propagator
    /// let entity = world.spawn(MyData(0)).id();
    ///
    /// let signal = Signal::from_component::<MyData>(entity)
    ///     .map(|In(d): In<MyData>| { println!("Data: {}", d.0); TERMINATE });
    ///
    /// // Register the signal
    /// let handle = signal.register(&mut world);
    ///
    /// // Later, cleanup the signal
    /// handle.cleanup(&mut world);
    ///
    /// // The systems associated with the handle should now be despawned.
    /// // Verification would require checking world state, complex for a doctest.
    /// ```
    pub fn cleanup(self, world: &mut World) {
        if let Some(mut propagator) = world.remove_resource::<SignalPropagator>() {
            propagator.remove_node(world, self.0);
            world.insert_resource(propagator);
        } else {
            warn!(
                "SignalPropagator not found during cleanup for system {:?}",
                self.0
            );
            if let Ok(entity_commands) = world.get_entity_mut(self.0) {
                warn!(
                    "Despawning entity {:?} directly as SignalPropagator was missing.",
                    self.0
                );
                entity_commands.despawn();
            }
        }
    }
}

/// A declarative builder for creating signal chains.
///
/// `Signal<I, O>` represents a computation that conceptually transforms
/// signals of type `I` into signals of type `O`. However, `I` is often `()`
/// for source signals.
///
/// Use the `from_*` static methods (e.g., [`Signal::from_component`]) to start a chain,
/// then chain combinators like [`map`](Signal::map) and [`combine_with`](Signal::combine_with).
/// Finally, call [`register`](Signal::register) to activate the signal chain by creating
/// the necessary Bevy systems.
///
/// Signals are `Clone`, allowing the description of a chain to be reused or passed around
/// before registration. Registration itself (`register`) does not consume the `Signal`
/// due to the internal use of `Arc`. Registering the same `Signal` multiple times is safe
/// (it won't duplicate systems or connections) but will return distinct [`SignalHandle`]s
/// pointing to the same underlying final system.
#[derive(Clone)]
pub struct Signal<I: Send + Sync + 'static, O: Send + Sync + 'static> {
    /// Internal implementation detail holding the actual signal node definition.
    /// This uses `Arc<dyn ISignal>` for type erasure and shared ownership.
    pub(crate) signal_impl: Arc<dyn ISignal>,
    /// Marker for input/output types, not used functionally but helps with type inference.
    pub(crate) _marker: PhantomData<fn() -> (I, O)>,
}

/// Helper to create a source signal node. Wraps the registration function
/// in the `SourceSignal` struct and creates the public `Signal` type.
pub(crate) fn create_source<O>(
    register_fn: Arc<dyn Fn(&mut World) -> UntypedSystemId + Send + Sync + 'static>,
) -> Signal<(), O>
where
    O: Send + Sync + 'static,
{
    let source_signal = SourceSignal { register_fn };
    Signal {
        signal_impl: Arc::new(source_signal),
        _marker: PhantomData,
    }
}

// Static methods to start signal chains
impl Signal<(), ()> {
    /// Starts a signal chain from an arbitrary Bevy system.
    ///
    /// The provided `system` must take `In<()>` and return `Option<O>`.
    /// It will be registered as a "root" signal, meaning its execution is triggered
    /// automatically by the [`SignalPropagator`] each update cycle.
    ///
    /// The signal emits values of type `O` whenever the system returns `Some(O)`.
    ///
    /// The system needs to be `Clone` because the `Signal` itself is `Clone`.
    ///
    /// ```no_run
    /// use bevy::prelude::*;
    /// use jonmo::*;
    ///
    /// fn my_source_system(_: In<()>) -> Option<i32> {
    ///     Some(42)
    /// }
    ///
    /// let signal = Signal::from_system(my_source_system);
    ///
    /// // To activate, register it:
    /// // let mut world = World::new();
    /// // world.init_resource::<SignalPropagator>();
    /// // let handle = signal.register(&mut world);
    /// ```
    pub fn from_system<O, M>(
        system: impl IntoSystem<In<()>, Option<O>, M> + Send + Sync + Clone + 'static,
    ) -> Signal<(), O>
    where
        O: FromReflect + Send + Sync + 'static,
    {
        let register_fn = Arc::new(move |world: &mut World| {
            let system_id = register_signal(world, system.clone());
            mark_signal_root(world, system_id.entity());
            system_id.entity()
        });
        create_source(register_fn)
    }

    /// Starts a signal chain that emits an `Entity` ID once.
    ///
    /// This is often used as a starting point for chains that then query components
    /// on that specific entity using [`map`](Signal::map).
    ///
    /// Internally uses [`entity_root`] and [`Signal::from_system`].
    ///
    /// ```no_run
    /// use bevy::prelude::*;
    /// use jonmo::*;
    ///
    /// // Assume 'my_entity' is an Entity created elsewhere
    /// let my_entity = Entity::PLACEHOLDER;
    /// let signal = Signal::from_entity(my_entity);
    ///
    /// // To activate, register it:
    /// // let mut world = World::new();
    /// // world.init_resource::<SignalPropagator>();
    /// // let handle = signal.register(&mut world);
    /// ```
    pub fn from_entity(entity: Entity) -> Signal<(), Entity> {
        Self::from_system(entity_root(entity))
    }

    /// Starts a signal chain that emits when a specific component `C` on a given `entity` changes.
    ///
    /// The signal emits the new value of the component `C` whenever Bevy's change detection
    /// (`Changed<C>`) detects a change for that specific entity.
    ///
    /// Requires the component `C` to implement `Component`, `FromReflect`, `Clone`, `Send`, `Sync`, and `'static`.
    ///
    /// ```no_run
    /// use bevy::prelude::*;
    /// use jonmo::*;
    ///
    /// #[derive(Component, Reflect, Clone, Default, PartialEq)]
    /// #[reflect(Component)]
    /// struct MyComponent(i32);
    ///
    /// // Assume 'my_entity' is an Entity with MyComponent created elsewhere
    /// let my_entity = Entity::PLACEHOLDER;
    /// let signal = Signal::from_component::<MyComponent>(my_entity);
    ///
    /// // To activate, register it:
    /// // let mut world = World::new();
    /// // world.init_resource::<SignalPropagator>();
    /// // let handle = signal.register(&mut world);
    /// ```
    pub fn from_component<C>(entity: Entity) -> Signal<(), C>
    where
        C: Component + FromReflect + Clone + Send + Sync + 'static,
    {
        let component_query_system =
            move |_: In<()>, query: Query<&'static C, Changed<C>>| query.get(entity).ok().cloned();
        Self::from_system(component_query_system)
    }

    /// Starts a signal chain that emits when a specific resource `R` changes.
    ///
    /// The signal emits a clone of the resource `R` whenever Bevy's change detection
    /// (`Res<R>::is_changed()`) detects a change.
    ///
    /// Requires the resource `R` to implement `Resource`, `FromReflect`, `Clone`, `Send`, `Sync`, and `'static`.
    ///
    /// ```no_run
    /// use bevy::prelude::*;
    /// use jonmo::*;
    ///
    /// #[derive(Resource, Reflect, Clone, Default, PartialEq)]
    /// #[reflect(Resource)]
    /// struct MyResource(f32);
    ///
    /// let signal = Signal::from_resource::<MyResource>();
    ///
    /// // To activate, register it:
    /// // let mut world = World::new();
    /// // world.init_resource::<SignalPropagator>();
    /// // world.init_resource::<MyResource>();
    /// // let handle = signal.register(&mut world);
    /// ```
    pub fn from_resource<R>() -> Signal<(), R>
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

// Combinator methods
impl<I, O> Signal<I, O>
where
    I: Send + Sync + 'static,
    O: FromReflect + Send + Sync + 'static,
{
    /// Appends a transformation step to the signal chain using a Bevy system.
    ///
    /// The provided `system` takes the output `O` of the previous step as `In<O>`
    /// and should return `Option<U>`.
    /// - If the system returns `Some(U)`, the value `U` is propagated to the next step in the graph.
    /// - If the system returns `None`, the signal chain terminates *along this path* for the current
    ///   frame's execution (see [`TERMINATE`]). Propagation stops here.
    ///
    /// The system needs to be `Clone` because the `Signal` itself is `Clone`.
    ///
    /// Common uses include:
    /// - Querying data based on the input signal (e.g., querying components using an `Entity` signal).
    /// - Transforming the data type.
    /// - Filtering signals by returning `None`.
    /// - Performing side effects (e.g., using `Commands`).
    ///
    /// ```no_run
    /// use bevy::prelude::*;
    /// use jonmo::*;
    ///
    /// fn multiply_by_two(In(val): In<i32>) -> Option<i32> {
    ///     Some(val * 2)
    /// }
    ///
    /// fn print_and_terminate(In(val): In<i32>) {
    ///     println!("Final value: {}", val);
    ///     // Implicitly returns None / TERMINATE, stopping further propagation.
    /// }
    ///
    /// let signal = Signal::from_system(|| Some(5)) // Source emits 5
    ///     .map(multiply_by_two) // Emits 10
    ///     .map(print_and_terminate); // Prints 10 and terminates this path
    ///
    /// // To activate, register it:
    /// // let mut world = World::new();
    /// // world.init_resource::<SignalPropagator>();
    /// // let handle = signal.register(&mut world);
    /// ```
    pub fn map<U, M>(
        self,
        system: impl IntoSystem<In<O>, Option<U>, M> + Send + Sync + Clone + 'static,
    ) -> Signal<I, U>
    where
        U: FromReflect + Send + Sync + 'static,
    {
        let register_fn = Arc::new(move |world: &mut World, prev_id_entity: UntypedSystemId| {
            let new_system_id = register_signal(world, system.clone());
            pipe_signal(world, prev_id_entity, new_system_id.entity());
            new_system_id.entity()
        });

        let map_signal = MapSignal {
            prev_signal: self.signal_impl.clone(),
            register_fn,
        };

        Signal {
            signal_impl: Arc::new(map_signal),
            _marker: PhantomData,
        }
    }
}

impl<I, O> Signal<I, O>
where
    I: Send + Sync + 'static,
    O: FromReflect + GetTypeRegistration + Typed + Send + Sync + Clone + 'static + std::fmt::Debug,
{
    /// Combines this signal chain (`self`) with another signal chain (`other`).
    ///
    /// This method creates a new signal that waits until *both* `self` and `other`
    /// have produced at least one value. Once both have produced a value, it emits
    /// a tuple `(O, O2)` containing the *most recent* value from each chain.
    ///
    /// After emitting, it resets and waits for the next pair of values from both branches.
    /// If one branch produces multiple values while waiting for the other, only the
    /// latest value from the faster branch is used when the combination finally occurs.
    ///
    /// Requires the output types `O` and `O2` to implement necessary reflection traits
    /// (`FromReflect`, `GetTypeRegistration`, `Typed`) and `Clone` for internal caching.
    ///
    /// ```no_run
    /// use bevy::prelude::*;
    /// use jonmo::*;
    ///
    /// #[derive(Resource, Reflect, Clone, Default, PartialEq, Debug)]
    /// #[reflect(Resource)]
    /// struct SourceA(i32);
    ///
    /// #[derive(Resource, Reflect, Clone, Default, PartialEq, Debug)]
    /// #[reflect(Resource)]
    /// struct SourceB(String);
    ///
    /// let signal_a = Signal::from_resource::<SourceA>();
    /// let signal_b = Signal::from_resource::<SourceB>();
    ///
    /// let combined_signal = signal_a
    ///     .combine_with(signal_b)
    ///     .map(|In((a, b)): In<(SourceA, SourceB)>| {
    ///         println!("Combined: {:?}, {:?}", a, b);
    ///         TERMINATE
    ///     });
    ///
    /// // To activate, register it:
    /// // let mut world = World::new();
    /// // world.init_resource::<SignalPropagator>();
    /// // world.init_resource::<SourceA>();
    /// // world.init_resource::<SourceB>();
    /// // let handle = combined_signal.register(&mut world);
    /// ```
    pub fn combine_with<I2, O2>(self, other: Signal<I2, O2>) -> Signal<(), (O, O2)>
    where
        I2: Send + Sync + 'static,
        O2: FromReflect
            + GetTypeRegistration
            + Typed
            + Send
            + Sync
            + Clone
            + 'static
            + std::fmt::Debug,
    {
        let register_fn = Arc::new(
            move |world: &mut World,
                  left_id_entity: UntypedSystemId,
                  right_id_entity: UntypedSystemId| {
                let combined_id = combine_signal::<O, O2>(world, left_id_entity, right_id_entity);
                combined_id.entity()
            },
        );

        let combine_signal = CombineSignal {
            left_signal: self.signal_impl.clone(),
            right_signal: other.signal_impl.clone(),
            register_fn,
        };

        Signal {
            signal_impl: Arc::new(combine_signal),
            _marker: PhantomData,
        }
    }
}

impl<I, O> Signal<I, O>
where
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
{
    /// Registers all systems involved in this signal chain within the Bevy `World`.
    ///
    /// This activates the signal chain. It traverses the chain definition, registers
    /// each system, connects them in the internal [`SignalPropagator`] graph, and marks
    /// root systems.
    ///
    /// Returns a [`SignalHandle`] which can be used later to [`cleanup`](SignalHandle::cleanup)
    /// the registered systems and connections associated with this specific chain instance.
    ///
    /// **Note:** This method takes `&self` and does not consume the `Signal`. Calling
    /// `register` multiple times on the same `Signal` or clones of it is safe (systems
    /// and connections are not duplicated), but it will return new `SignalHandle`s
    /// pointing to the same underlying final system entity. Cleaning up one handle
    /// will affect the graph shared by all handles derived from the same registration(s).
    ///
    /// ```no_run
    /// use bevy::prelude::*;
    /// use jonmo::*;
    ///
    /// let signal = Signal::from_system(|| Some(1))
    ///     .map(|In(x): In<i32>| Some(x + 1));
    ///
    /// let mut world = World::new();
    /// world.init_resource::<SignalPropagator>(); // Plugin normally does this
    ///
    /// // Register the signal chain
    /// let handle: SignalHandle = signal.register(&mut world);
    ///
    /// // The handle identifies the last system in the chain
    /// println!("Registered signal with handle: {:?}", handle);
    ///
    /// // You can now use the handle to clean up later if needed
    /// // handle.cleanup(&mut world);
    /// ```
    pub fn register(&self, world: &mut World) -> SignalHandle {
        let last_system_id = self.signal_impl.register(world);
        SignalHandle(last_system_id)
    }
}

/// System that drives signal propagation by calling [`SignalPropagator::execute`].
/// Added to the `Update` schedule by the [`JonmoPlugin`]. This system runs once per frame.
pub(crate) fn process_signals(world: &mut World) {
    if let Some(propagator) = world.remove_resource::<SignalPropagator>() {
        propagator.execute(world);
        world.insert_resource(propagator);
    }
}

/// The Bevy plugin required for `jonmo` signals to function.
///
/// Adds the necessary [`SignalPropagator`] resource and the system that drives
/// signal propagation (`process_signals`) to the `Update` schedule.
///
/// ```no_run
/// use bevy::prelude::*;
/// use jonmo::JonmoPlugin;
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

#[cfg(test)]
mod tests {
    use super::*; // Import items from the library root
    use std::sync::{Arc, Mutex};

    // Helper component to track if a system ran
    #[derive(Component, Default, Clone, Debug, PartialEq)]
    struct Ran(bool);

    // Helper component for component-based signals
    #[derive(Component, Reflect, Clone, Default, PartialEq, Debug)]
    #[reflect(Component)]
    struct TestValue(i32);

    // Helper resource for resource-based signals
    #[derive(Resource, Reflect, Clone, Default, PartialEq, Debug)]
    #[reflect(Resource)]
    struct TestResource(String);

    // Helper resource to store results for assertions
    #[derive(Resource, Default)]
    struct TestResult<T: Send + Sync + 'static>(Arc<Mutex<Option<T>>>);

    impl<T: Send + Sync + 'static> TestResult<T> {
        fn set(&self, value: T) {
            *self.0.lock().unwrap() = Some(value);
        }
        // fn get(&self) -> Option<T> // Not used in current tests
        // where
        //     T: Clone,
        // {
        //     self.0.lock().unwrap().clone()
        // }
        fn take(&self) -> Option<T> {
            self.0.lock().unwrap().take()
        }
    }

    // Helper resource to store vec results for assertions
    #[derive(Resource, Default)]
    struct TestVecResult<T: Send + Sync + 'static>(Arc<Mutex<Vec<T>>>);

    impl<T: Send + Sync + 'static> TestVecResult<T> {
        fn push(&self, value: T) {
            self.0.lock().unwrap().push(value);
        }
        fn get_cloned(&self) -> Vec<T>
        where
            T: Clone,
        {
            self.0.lock().unwrap().clone()
        }
    }

    // Helper resource to store counter results for assertions
    #[derive(Resource, Default)]
    struct TestCounterResult(Arc<Mutex<i32>>);

    impl TestCounterResult {
        fn increment(&self, value: i32) {
            *self.0.lock().unwrap() += value;
        }
        fn get(&self) -> i32 {
            *self.0.lock().unwrap()
        }
    }

    // --- Test Setup ---

    fn setup_app() -> App {
        let mut app = App::new();
        // Use MinimalPlugins for lighter testing if DefaultPlugins are too heavy
        // app.add_plugins(MinimalPlugins);
        app.add_plugins(JonmoPlugin);
        // Register types needed for tests
        app.register_type::<TestValue>();
        app.register_type::<TestResource>();
        app
    }

    // --- Tests ---

    #[test]
    fn test_basic_map_chain() {
        let mut app = setup_app();
        let result_store = Arc::new(Mutex::new(None::<i32>));
        app.insert_resource(TestResult(result_store.clone()));

        let signal = Signal::from_system(|| Some(10))
            .map(|In(v): In<i32>| Some(v * 2))
            .map(|In(v): In<i32>, res: Res<TestResult<i32>>| {
                res.set(v);
                TERMINATE
            });

        let _handle = signal.register(&mut app.world);

        app.update(); // Run process_signals

        assert_eq!(result_store.lock().unwrap().take(), Some(20));
    }

    #[test]
    fn test_from_component() {
        let mut app = setup_app();
        let result_store = Arc::new(Mutex::new(None::<i32>));
        app.insert_resource(TestResult(result_store.clone()));

        let entity = app.world.spawn(TestValue(5)).id();

        let signal = Signal::from_component::<TestValue>(entity).map(
            |In(v): In<TestValue>, res: Res<TestResult<i32>>| {
                res.set(v.0);
                TERMINATE
            },
        );

        let _handle = signal.register(&mut app.world);

        // Initial run - component exists but hasn't "changed" yet relative to signal start
        app.update();
        assert_eq!(result_store.lock().unwrap().take(), None);

        // Modify the component
        app.world
            .entity_mut(entity)
            .get_mut::<TestValue>()
            .unwrap()
            .0 = 15;

        // Run again - change should be detected
        app.update();
        assert_eq!(result_store.lock().unwrap().take(), Some(15));

        // Run again - no change
        app.update();
        assert_eq!(result_store.lock().unwrap().take(), None);
    }

    #[test]
    fn test_from_resource() {
        let mut app = setup_app();
        let result_store = Arc::new(Mutex::new(None::<String>));
        app.insert_resource(TestResult(result_store.clone()));
        app.init_resource::<TestResource>();

        let signal = Signal::from_resource::<TestResource>().map(
            |In(v): In<TestResource>, res: Res<TestResult<String>>| {
                res.set(v.0);
                TERMINATE
            },
        );

        let _handle = signal.register(&mut app.world);

        // Initial run - resource exists but hasn't "changed" yet relative to signal start
        app.update();
        assert_eq!(result_store.lock().unwrap().take(), None);

        // Modify the resource
        app.world.resource_mut::<TestResource>().0 = "hello".to_string();

        // Run again - change should be detected
        app.update();
        assert_eq!(
            result_store.lock().unwrap().take(),
            Some("hello".to_string())
        );

        // Run again - no change
        app.update();
        assert_eq!(result_store.lock().unwrap().take(), None);
    }

    #[test]
    fn test_dedupe() {
        let mut app = setup_app();
        let result_store = Arc::new(Mutex::new(Vec::<i32>::new()));
        app.insert_resource(TestVecResult(result_store.clone()));

        let entity = app.world.spawn(TestValue(1)).id();

        let signal = Signal::from_component::<TestValue>(entity)
            .map(dedupe) // Add dedupe here
            .map(|In(v): In<TestValue>, res: Res<TestVecResult<i32>>| {
                res.push(v.0);
                TERMINATE
            });

        let _handle = signal.register(&mut app.world);

        // Initial change
        app.world
            .entity_mut(entity)
            .get_mut::<TestValue>()
            .unwrap()
            .0 = 10;
        app.update();
        assert_eq!(result_store.lock().unwrap().clone(), vec![10]);

        // No change
        app.update();
        assert_eq!(result_store.lock().unwrap().clone(), vec![10]); // Should not run again

        // Change again
        app.world
            .entity_mut(entity)
            .get_mut::<TestValue>()
            .unwrap()
            .0 = 20;
        app.update();
        assert_eq!(result_store.lock().unwrap().clone(), vec![10, 20]);

        // Change back
        app.world
            .entity_mut(entity)
            .get_mut::<TestValue>()
            .unwrap()
            .0 = 10;
        app.update();
        assert_eq!(result_store.lock().unwrap().clone(), vec![10, 20, 10]);

        // No change
        app.update();
        assert_eq!(result_store.lock().unwrap().clone(), vec![10, 20, 10]); // Should not run again
    }

    #[test]
    fn test_combine_with() {
        let mut app = setup_app();
        let result_store = Arc::new(Mutex::new(None::<(i32, String)>));
        app.insert_resource(TestResult(result_store.clone()));
        app.init_resource::<TestResource>(); // Source B

        let entity_a = app.world.spawn(TestValue(0)).id(); // Source A

        let signal_a = Signal::from_component::<TestValue>(entity_a).map(dedupe);
        let signal_b = Signal::from_resource::<TestResource>().map(dedupe);

        let combined = signal_a.combine_with(signal_b).map(
            |In((a, b)): In<(TestValue, TestResource)>, res: Res<TestResult<(i32, String)>>| {
                res.set((a.0, b.0));
                TERMINATE
            },
        );

        let _handle = combined.register(&mut app.world);

        // 1. Initial state - nothing changed yet relative to signals
        app.update();
        assert_eq!(result_store.lock().unwrap().take(), None);

        // 2. Change A
        app.world
            .entity_mut(entity_a)
            .get_mut::<TestValue>()
            .unwrap()
            .0 = 1;
        app.update();
        assert_eq!(result_store.lock().unwrap().take(), None); // B hasn't fired yet

        // 3. Change B
        app.world.resource_mut::<TestResource>().0 = "first".to_string();
        app.update();
        // Now both have fired, combine should trigger with (1, "first")
        assert_eq!(
            result_store.lock().unwrap().take(),
            Some((1, "first".to_string()))
        );

        // 4. Change B again
        app.world.resource_mut::<TestResource>().0 = "second".to_string();
        app.update();
        assert_eq!(result_store.lock().unwrap().take(), None); // A hasn't fired again yet

        // 5. Change A again
        app.world
            .entity_mut(entity_a)
            .get_mut::<TestValue>()
            .unwrap()
            .0 = 2;
        app.update();
        // Combine should trigger with (2, "second") - uses latest B
        assert_eq!(
            result_store.lock().unwrap().take(),
            Some((2, "second".to_string()))
        );

        // 6. Change A multiple times, then B
        app.world
            .entity_mut(entity_a)
            .get_mut::<TestValue>()
            .unwrap()
            .0 = 3;
        app.update();
        app.world
            .entity_mut(entity_a)
            .get_mut::<TestValue>()
            .unwrap()
            .0 = 4;
        app.update();
        app.world.resource_mut::<TestResource>().0 = "third".to_string();
        app.update();
        // Combine should trigger with latest A (4) and latest B ("third")
        assert_eq!(
            result_store.lock().unwrap().take(),
            Some((4, "third".to_string()))
        );
    }

    #[test]
    fn test_cleanup() {
        let mut app = setup_app();
        let result_store = Arc::new(Mutex::new(0)); // Count how many times the last system runs
        app.insert_resource(TestCounterResult(result_store.clone()));

        let signal =
            Signal::from_system(|| Some(1)).map(|In(v): In<i32>, res: Res<TestCounterResult>| {
                res.increment(v);
                Some(v) // Pass through
            });

        let handle = signal.register(&mut app.world);
        let system_entity = handle.0; // Get the entity ID

        // Check system exists and runs
        assert!(app.world.get_entity(system_entity).is_some());
        assert!(app.world.get::<SystemRunner>(system_entity).is_some());
        app.update();
        assert_eq!(result_store.lock().unwrap().clone(), 1);

        // Cleanup
        handle.cleanup(&mut app.world);

        // Check system entity is despawned
        assert!(app.world.get_entity(system_entity).is_none());

        // Check propagator no longer contains the node (requires access, tricky)
        // We can indirectly test by seeing if the system runs again
        app.update();
        assert_eq!(result_store.lock().unwrap().clone(), 1); // Count should not increase

        // Check trying to cleanup again doesn't panic
        handle.cleanup(&mut app.world);
    }
}
