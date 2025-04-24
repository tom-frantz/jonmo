use crate::tree::*;
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{entity, prelude::*, system::SystemId};
use bevy_hierarchy::{BuildChildren, DespawnRecursiveExt};
use bevy_log::prelude::*;
use bevy_reflect::{FromReflect, GetTypeRegistration, PartialReflect, Typed};
use std::{
    any::TypeId,
    marker::PhantomData,
    sync::{Arc, Mutex},
};

#[derive(Clone)]
pub(crate) enum RegisterOnceSignal {
    // the function is just indirection required because IntoSystem isn't dyn compatible
    System(Arc<Mutex<Option<Box<dyn FnOnce(&mut World) -> SignalSystem + Send + Sync + 'static>>>>),
    Registered(SignalSystem),
}

impl RegisterOnceSignal {
    /// Registers the system if it hasn't been registered yet.
    /// Returns the system ID of the registered system.
    pub fn register(&mut self, world: &mut World) -> SignalSystem {
        match self {
            RegisterOnceSignal::System(f) => {
                let signal = f.lock().unwrap().take().unwrap()(world).into();
                *self = RegisterOnceSignal::Registered(signal);
                signal
            }
            RegisterOnceSignal::Registered(signal) => {
                if let Ok(mut system) = world.get_entity_mut(**signal) {
                    if let Some(mut ref_count) = system.get_mut::<SignalReferenceCount>() {
                        ref_count.increment();
                    }
                }
                *signal
            }
        }
    }
}

/// Type alias for the boxed combine registration function.
type CombineRegisterFn<O1, O2> = dyn Fn(
        &mut World,
        SignalSystem,
        SignalSystem,
    ) -> (
        SystemId<In<(Option<O1>, Option<O2>)>, Option<(O1, O2)>>,
        Vec<SignalSystem>,
    ) + Send
    + Sync
    + 'static;

/// Type alias for the boxed map registration function.
type MapRegisterFn = dyn FnOnce(
        &mut World,
        SignalSystem, // Previous system's entity ID
    ) -> SignalSystem // Registered map system's entity ID
    + Send
    + Sync
    + 'static;

// --- Consolidated Signal Trait ---

/// Represents a value that changes over time and handles internal registration logic.
///
/// Signals are the core building block for reactive data flow. They are typically
/// created using methods on the [`SignalBuilder`] struct (e.g., [`SignalBuilder::from_component`])
/// and then transformed or combined using methods from the [`SignalExt`] trait.
/// This trait combines the public concept of a signal with its internal registration mechanism.
pub trait Signal: Send + Sync + 'static {
    /// The type of value produced by this signal.
    type Item: Send + Sync + 'static;

    /// Registers the systems associated with this node and its predecessors in the `World`.
    /// Returns a [`SignalHandle`] containing the entities of *all* systems
    /// registered or reference-counted during this specific registration call instance.
    /// **Note:** This method is intended for internal use by the signal combinators and registration process.
    fn register_signal(self, world: &mut World) -> SignalHandle; // Changed return type
}

// --- Signal Node Structs ---

/// Struct representing a source node in the signal chain definition. Implements [`Signal`].
#[derive(Clone)]
pub struct Source<O>
where
    O: Send + Sync + 'static,
{
    signal: RegisterOnceSignal,
    _marker: PhantomData<O>,
}

impl<O> Signal for Source<O>
where
    O: Clone + Send + Sync + 'static,
{
    type Item = O;

    fn register_signal(mut self, world: &mut World) -> SignalHandle {
        SignalHandle::new(vec![self.signal.register(world)])
    }
}

/// Struct representing a map node in the signal chain definition. Implements [`Signal`].
/// Generic only over the previous signal (`Prev`) and the output type (`U`).
#[derive(Clone)]
pub struct Map<Upstream, O>
where
    Upstream: Signal,
    Upstream::Item: FromReflect + Send + Sync + 'static,
    O: Send + Sync + 'static,
{
    pub(crate) upstream: Upstream,
    pub(crate) signal: RegisterOnceSignal,
    _marker: PhantomData<O>,
}

impl<Upstream, O> Signal for Map<Upstream, O>
where
    Upstream: Signal,
    Upstream::Item: FromReflect + Send + Sync + 'static,
    O: FromReflect + Send + Sync + 'static,
{
    type Item = O;

    fn register_signal(mut self, world: &mut World) -> SignalHandle {
        let SignalHandle(mut lineage) = self.upstream.register_signal(world);
        let signal = self.signal.register(world);
        if let Some(&parent) = lineage.last() {
            pipe_signal(world, parent, signal);
        } else {
            error!("Map signal parent registration returned empty ID list.");
        }
        lineage.push(signal);
        SignalHandle::new(lineage)
    }
}

#[derive(Clone)]
pub struct MapComponent<Upstream, C>
where
    Upstream: Signal<Item = Entity>,
    Upstream::Item: Send + Sync + 'static,
    C: Component + Clone + Send + Sync + 'static,
{
    pub(crate) upstream: Upstream,
    _marker: PhantomData<C>,
}

impl<Upstream, C> Signal for MapComponent<Upstream, C>
where
    Upstream: Signal<Item = Entity>,
    Upstream::Item: FromReflect + Send + Sync + 'static,
    C: Component + Clone + FromReflect + Send + Sync + 'static,
{
    type Item = C;

    fn register_signal(self, world: &mut World) -> SignalHandle {
        let SignalHandle(mut lineage) = self.upstream.register_signal(world);
        let signal = register_signal::<_, C, _, _, _>(
            world,
            |In(entity): In<Entity>, components: Query<&C>| components.get(entity).ok().cloned(),
        );
        if let Some(&parent) = lineage.last() {
            pipe_signal(world, parent, signal);
        } else {
            error!("MapComponent signal parent registration returned empty ID list.");
        }
        lineage.push(signal);
        SignalHandle::new(lineage)
    }
}

/// Struct representing a combine node in the signal chain definition. Implements [`Signal`].
#[derive(Clone)]
pub struct Combine<Left, Right>
where
    Left: Signal,
    Right: Signal,
    Left::Item: FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static,
    Right::Item: FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static,
{
    pub(crate) left: Left,
    pub(crate) right: Right,
    _marker: PhantomData<(Left::Item, Right::Item)>,
}

impl<Left, Right> Signal for Combine<Left, Right>
where
    Left: Signal,
    Left::Item: FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static,
    Right: Signal,
    Right::Item: FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static,
{
    type Item = (Left::Item, Right::Item);

    fn register_signal(self, world: &mut World) -> SignalHandle {
        let combiner = register_signal::<_, (Left::Item, Right::Item), _, _, _>(
            world,
            move |In((left_option, right_option)): In<(
                Option<Left::Item>,
                Option<Right::Item>,
            )>,
                  mut left_cache: Local<Option<Left::Item>>,
                  mut right_cache: Local<Option<Right::Item>>| {
                if left_option.is_some() {
                    *left_cache = left_option;
                }
                if right_option.is_some() {
                    *right_cache = right_option;
                }
                if left_cache.is_some() && right_cache.is_some() {
                    left_cache.take().zip(right_cache.take())
                } else {
                    None
                }
            },
        );
        let left_wrapper = register_signal(world, |In(left_val): In<Left::Item>| {
            (Some(left_val), None::<Right::Item>)
        });
        let right_wrapper = register_signal(world, |In(right): In<Right::Item>| {
            (None::<Left::Item>, Some(right))
        });
        let SignalHandle(mut left_lineage) = self.left.register_signal(world);
        if let Some(&parent) = left_lineage.last() {
            pipe_signal(world, parent, combiner);
        } else {
            error!("Combine left signal registration returned empty ID list.");
        }
        left_lineage.push(left_wrapper);
        let SignalHandle(mut right_lineage) = self.right.register_signal(world);
        if let Some(&parent) = right_lineage.last() {
            pipe_signal(world, parent, combiner);
        } else {
            error!("Combine right signal registration returned empty ID list.");
        }
        right_lineage.push(right_wrapper);
        left_lineage.append(&mut right_lineage);
        left_lineage.push(combiner);
        SignalHandle::new(left_lineage)
    }
}

/// Handle returned by [`SignalExt::register`] used for cleaning up the registered signal chain.
///
/// Contains the list of all system entities created by the specific `register` call
/// that produced this handle. Dropping the handle does *not* automatically clean up.
/// Use the [`cleanup`](SignalHandle::cleanup) method for explicit cleanup.
#[derive(Clone, Deref, DerefMut)]
pub struct SignalHandle(pub Vec<SignalSystem>);

impl SignalHandle {
    /// Creates a new SignalHandle.
    /// This is crate-public to allow construction from other modules.
    pub(crate) fn new(ids: Vec<SignalSystem>) -> Self {
        Self(ids)
    }

    /// Returns a slice containing the system IDs managed by this handle.
    pub(crate) fn system_ids(&self) -> &[SignalSystem] {
        &self.0
    }

    /// Decrements the reference count for each system associated with this handle.
    /// If a system's reference count reaches zero, it removes the system's node
    /// from the internal signal graph and despawns its associated entity from the Bevy `World`.
    pub fn cleanup(self, world: &mut World) {
        for &signal in &self.0 {
            if let Ok(mut entity) = world.get_entity_mut(*signal) {
                if let Some(mut ref_count) = entity.get_mut::<SignalReferenceCount>() {
                    if ref_count.decrement() == 0 {
                        entity.try_despawn_recursive();
                    }
                }
            }
        }
    }
}

pub(crate) fn register_once_signal_from_system<I, O, IOO, IS, M>(system: IS) -> RegisterOnceSignal
where
    I: FromReflect + Send + Sync + 'static,
    O: FromReflect + Send + Sync + 'static,
    IOO: Into<Option<O>> + Send + Sync + 'static,
    IS: IntoSystem<In<I>, IOO, M> + Send + Sync + 'static,
    M: Send + Sync + 'static,
{
    RegisterOnceSignal::System(Arc::new(Mutex::new(Some(Box::new(
        move |world: &mut World| {
            let system = world.register_system(system);
            let entity = system.entity();
            world.entity_mut(entity).insert((
                SignalReferenceCount::new(),
                SystemRunner {
                    runner: Arc::new(Box::new(move |world, input| {
                        match I::from_reflect(input.as_ref()) {
                            Some(input) => match world.run_system_with_input(system, input) {
                                Ok(output) => {
                                    if let Some(output) = output.into() {
                                        Some(Box::new(output) as Box<dyn PartialReflect>)
                                    } else {
                                        None // terminate
                                    }
                                }
                                Err(err) => {
                                    warn!("error running system {:?}: {}", system, err);
                                    None // terminate on error
                                }
                            },
                            None => {
                                warn!(
                                    "failed to downcast input for system {:?}<{:?}>: {:?}",
                                    system,
                                    input.reflect_type_path(),
                                    input
                                );
                                None
                            }
                        }
                    })),
                },
            ));
            system.into()
        },
    )))))
}

/// Provides static methods for creating new signal chains (source signals).
/// Use methods like [`SignalBuilder::from_component`] or [`SignalBuilder::from_system`]
/// to start building a signal chain.
pub struct SignalBuilder;

impl From<Entity> for Source<Entity> {
    fn from(entity: Entity) -> Self {
        SignalBuilder::from_entity(entity)
    }
}

// Static methods to start signal chains, now associated with SignalBuilder struct
impl SignalBuilder {
    pub fn from_system<O, IOO, IS, M>(system: IS) -> Source<O>
    where
        O: FromReflect + Send + Sync + 'static,
        IOO: Into<Option<O>> + Send + Sync + 'static,
        IS: IntoSystem<In<()>, IOO, M> + Send + Sync + 'static,
        M: Send + Sync + 'static,
    {
        Source {
            signal: register_once_signal_from_system(system),
            _marker: PhantomData,
        }
    }

    /// Creates a signal chain starting from a specific entity.
    ///
    /// The signal will emit the `Entity` ID whenever the propagation starts from this source.
    /// Useful for chains that operate on or react to changes related to this entity.
    /// Internally uses [`SignalBuilder::from_system`] with [`entity_root`].
    pub fn from_entity(entity: Entity) -> Source<Entity> {
        Self::from_system(move |_: In<()>| entity)
    }

    // /// Creates a signal chain that starts by observing changes to a specific component `C`
    // /// on a given `entity`.
    // ///
    // /// The signal emits the new value of the component `C` whenever it changes on the entity.
    // /// Requires the component `C` to implement `Component`, `FromReflect`, `Clone`, `Send`, `Sync`, and `'static`.
    // /// Internally uses [`SignalBuilder::from_system`].
    pub fn from_component<C>(entity: Entity) -> Source<C>
    where
        C: Component + FromReflect + GetTypeRegistration + Typed + Clone + Send + Sync + 'static,
    {
        Self::from_system(move |_: In<()>, components: Query<&C>| {
            components.get(entity).ok().cloned()
        })
    }

    /// Creates a signal chain that starts by observing changes to a specific resource `R`.
    ///
    /// The signal emits the new value of the resource `R` whenever it changes.
    /// Requires the resource `R` to implement `Resource`, `FromReflect`, `Clone`, `Send`, `Sync`, and `'static`.
    /// Internally uses [`SignalBuilder::from_system`].
    pub fn from_resource<R>() -> Source<R>
    where
        R: Resource + FromReflect + GetTypeRegistration + Typed + Clone + Send + Sync + 'static,
    {
        Self::from_system(move |_: In<()>, resource: Option<Res<R>>| {
            resource.filter(|r| r.is_changed()).map(|r| r.clone())
        })
    }
}

/// Extension trait providing combinator methods for types implementing [`Signal`] and [`Clone`].
pub trait SignalExt: Signal {
    // Remove SignalBuilderInternal bound
    /// Appends a transformation step to the signal chain using a Bevy system.
    ///
    /// The provided `system` takes the output `Item` of the previous step (wrapped in `In<Item>`)
    /// and returns an `Option<U>`. If it returns `Some(U)`, `U` is propagated to the next step.
    /// If it returns `None` (or [`TERMINATE`]), propagation along this branch stops for the frame.
    ///
    /// The system `F` must be `Clone` as it's captured for registration.
    /// Returns a [`Map`] signal node.
    fn map<O, IOO, IS, M>(self, system: IS) -> Map<Self, O>
    where
        Self: Sized,
        Self::Item: FromReflect + Send + Sync + 'static, // Use Signal::Item
        O: FromReflect + Send + Sync + 'static,
        IOO: Into<Option<O>> + Send + Sync + 'static,
        IS: IntoSystem<In<Self::Item>, IOO, M> // Use Signal::Item
            + Send
            + Sync
            + 'static,
        M: Send + Sync + 'static;

    fn map_component<C>(self) -> MapComponent<Self, C>
    where
        Self: Sized,
        Self: Signal<Item = Entity>,
        C: Component + Clone + FromReflect + Send + Sync + 'static;

    /// Combines this signal with another signal (`other`), producing a new signal that emits
    /// a tuple `(Self::Item, S2::Item)` of the outputs of both signals.
    ///
    /// The new signal emits a value only when *both* input signals have emitted at least one
    /// value since the last combined emission. It caches the latest value from each input.
    /// Both `self` and `other` must implement `Clone`.
    /// Returns a [`Combine`] signal node.
    fn combine<Other>(self, other: Other) -> Combine<Self, Other>
    where
        Self: Sized,
        Other: Signal,
        Self::Item:
            FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static + std::fmt::Debug,
        Other::Item:
            FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static + std::fmt::Debug;

    /// Registers all the systems defined in this signal chain into the Bevy `World`.
    ///
    /// This activates the signal chain. It traverses the internal representation (calling
    /// [`Signal::register`] recursively), registers each required Bevy system
    /// (or increments its reference count if already registered), connects them in the
    /// [`SignalPropagator`], and marks the source system(s) as roots.
    ///
    /// Returns a [`SignalHandle`] which can be used later to [`cleanup`](SignalHandle::cleanup)
    /// the systems created or referenced specifically by *this* `register` call.
    fn register(self, world: &mut World) -> SignalHandle;
}

// Implement SignalExt for any type T that implements Signal + Clone
impl<T> SignalExt for T
where
    T: Signal,
{
    fn map<O, IOO, IS, M>(self, system: IS) -> Map<Self, O>
    where
        T::Item: FromReflect + Send + Sync + 'static,
        O: FromReflect + Send + Sync + 'static,
        IOO: Into<Option<O>> + Send + Sync + 'static,
        IS: IntoSystem<In<T::Item>, IOO, M> + Send + Sync + 'static,
        M: Send + Sync + 'static,
    {
        Map {
            upstream: self,
            signal: register_once_signal_from_system(system),
            _marker: PhantomData,
        }
    }

    fn map_component<C>(self) -> MapComponent<Self, C>
    where
        Self: Sized,
        Self: Signal<Item = Entity>,
        C: Component + Clone + FromReflect + Send + Sync + 'static,
    {
        MapComponent {
            upstream: self,
            _marker: PhantomData,
        }
    }

    fn combine<Other>(self, other: Other) -> Combine<Self, Other>
    where
        Other: Signal,
        Self::Item:
            FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static + std::fmt::Debug,
        Other::Item:
            FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static + std::fmt::Debug,
    {
        Combine {
            left: self,
            right: other,
            _marker: PhantomData,
        }
    }

    fn register(self, world: &mut World) -> SignalHandle {
        T::register_signal(self, world)
    }
}
