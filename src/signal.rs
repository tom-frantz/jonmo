use crate::{tree::*, utils::SSs};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{entity, prelude::*, query::{QueryData, QueryFilter, WorldQuery}, system::{RunSystemOnce, SystemId}};
use bevy_hierarchy::{BuildChildren, DespawnRecursiveExt, HierarchyQueryExt, Parent};
use bevy_log::prelude::*;
use bevy_reflect::{FromReflect, GetTypeRegistration, PartialReflect, Reflect, TypePath, Typed};
use std::{
    any::TypeId, collections::VecDeque, fmt::Debug, marker::PhantomData, ops::Not, sync::{Arc, Mutex}
};

#[derive(Clone, Reflect)]
#[reflect(opaque)]
pub(crate) enum RegisterOnceSignal {
    // the function is just indirection required because IntoSystem isn't dyn compatible
    System(Arc<Mutex<Option<Box<dyn FnOnce(&mut World) -> SignalSystem + Send + Sync + 'static>>>>),
    Registered(SignalSystem),
}

impl RegisterOnceSignal {
    /// Registers the system if it hasn't been registered yet.
    /// Returns the system ID of the registered system.
    pub fn get(&mut self, world: &mut World) -> SignalSystem {
        match self {
            RegisterOnceSignal::System(f) => {
                let signal = f.lock().unwrap().take().unwrap()(world).into();
                *self = RegisterOnceSignal::Registered(signal);
                signal
            }
            RegisterOnceSignal::Registered(signal) => {
                if incr {
                    if let Ok(mut system) = world.get_entity_mut(**signal) {
                        if let Some(mut ref_count) = system.get_mut::<SignalReferenceCount>() {
                            ref_count.increment();
                        }
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
        SignalSystem, // Upstreamious system's entity ID
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
pub trait Signal: SSs {
    /// The type of value produced by this signal.
    type Item: SSs;

    /// Registers the systems associated with this node and its predecessors in the `World`.
    /// Returns a [`SignalHandle`] containing the entities of *all* systems
    /// registered or reference-counted during this specific registration call instance.
    /// **Note:** This method is intended for internal use by the signal combinators and registration process.
    fn register_signal(self, world: &mut World) -> SignalHandle; // Changed return type
}

// --- Signal Node Structs ---

/// Struct representing a source node in the signal chain definition. Implements [`Signal`].
#[derive(Clone, Reflect)]
pub struct Source<O>
where
    O: SSs,
{
    signal: RegisterOnceSignal,
    _marker: PhantomData<O>,
}

impl<O> Signal for Source<O>
where
    O: Clone + SSs,
{
    type Item = O;

    fn register_signal(mut self, world: &mut World) -> SignalHandle {
        SignalHandle::new(self.signal.get(world))
    }
}

/// Struct representing a map node in the signal chain definition. Implements [`Signal`].
/// Generic only over the previous signal (`Upstream`) and the output type (`U`).
#[derive(Clone)]
pub struct Map<Upstream, O>
where
    Upstream: Signal,
    Upstream::Item: FromReflect + SSs,
    O: SSs,
{
    pub(crate) upstream: Upstream,
    pub(crate) signal: RegisterOnceSignal,
    _marker: PhantomData<O>,
}

impl<Upstream, O> Signal for Map<Upstream, O>
where
    Upstream: Signal,
    Upstream::Item: FromReflect + SSs,
    O: FromReflect + SSs,
{
    type Item = O;

    fn register_signal(mut self, world: &mut World) -> SignalHandle {
        let SignalHandle(upstream) = self.upstream.register(world);
        let signal = self.signal.get(world);
        pipe_signal(world, upstream, signal);
        SignalHandle::new(signal)
    }
}

#[derive(Clone)]
pub struct MapComponent<Upstream, C>
where
    Upstream: Signal<Item = Entity>,
    Upstream::Item: SSs,
    C: Component + Clone + SSs,
{
    pub(crate) signal: Map<Upstream, C>,
}

impl<Upstream, C> Signal for MapComponent<Upstream, C>
where
    Upstream: Signal<Item = Entity>,
    Upstream::Item: FromReflect + SSs,
    C: Component + Clone + FromReflect + SSs,
{
    type Item = C;

    fn register_signal(self, world: &mut World) -> SignalHandle {
        self.signal.register(world)
    }
}

pub struct Dedupe<Upstream>
where
    Upstream: Signal,
    Upstream::Item: FromReflect + SSs,
{
    pub(crate) signal: Map<Upstream, Upstream::Item>,
}

impl<Upstream> Signal for Dedupe<Upstream>
where
    Upstream: Signal,
    Upstream::Item: FromReflect + SSs,
{
    type Item = Upstream::Item;

    fn register_signal(self, world: &mut World) -> SignalHandle {
        self.signal.register(world)
    }
}

pub struct First<Upstream>
where
    Upstream: Signal,
    Upstream::Item: FromReflect + SSs,
{
    pub(crate) signal: Map<Upstream, Upstream::Item>,
}

impl<Upstream> Signal for First<Upstream>
where
    Upstream: Signal,
    Upstream::Item: FromReflect + SSs,
{
    type Item = Upstream::Item;

    fn register_signal(self, world: &mut World) -> SignalHandle {
        self.signal.register(world)
    }
}

pub struct SignalDebug<Upstream>
where
    Upstream: Signal,
    Upstream::Item: FromReflect + SSs,
{
    pub(crate) signal: Map<Upstream, Upstream::Item>,
}

impl<Upstream> Signal for SignalDebug<Upstream>
where
    Upstream: Signal,
    Upstream::Item: FromReflect + SSs,
{
    type Item = Upstream::Item;

    fn register_signal(self, world: &mut World) -> SignalHandle {
        self.signal.register(world)
    }
}

/// Struct representing a combine node in the signal chain definition. Implements [`Signal`].
#[derive(Clone)]
pub struct Combine<Left, Right>
where
    Left: Signal,
    Right: Signal,
    Left::Item: FromReflect + GetTypeRegistration + Typed + SSs,
    Right::Item: FromReflect + GetTypeRegistration + Typed + SSs,
{
    pub(crate) left_wrapper: Map<Left, (Option<Left::Item>, Option<Right::Item>)>,
    pub(crate) right_wrapper: Map<Right, (Option<Left::Item>, Option<Right::Item>)>,
    pub(crate) signal: RegisterOnceSignal,
}

impl<Left, Right> Signal for Combine<Left, Right>
where
    Left: Signal,
    Left::Item: FromReflect + GetTypeRegistration + Typed + SSs,
    Right: Signal,
    Right::Item: FromReflect + GetTypeRegistration + Typed + SSs,
{
    type Item = (Left::Item, Right::Item);

    fn register_signal(mut self, world: &mut World) -> SignalHandle {
        let SignalHandle(left_upstream) = self.left_wrapper.register(world);
        let SignalHandle(right_upstream) = self.right_wrapper.register(world);
        let signal = self.signal.get(world);
        pipe_signal(world, left_upstream, signal);
        pipe_signal(world, right_upstream, signal);
        SignalHandle::new(signal)
    }
}

struct Flatten<Upstream>
where
    Upstream: Signal,
    Upstream::Item: FromReflect + Signal,
    <Upstream::Item as Signal>::Item: FromReflect + SSs,
{
    pub(crate) signal: Map<Upstream, <Upstream::Item as Signal>::Item>,
}

impl<Upstream> Signal for Flatten<Upstream>
where
    Upstream: Signal,
    Upstream::Item: FromReflect + Signal,
    <Upstream::Item as Signal>::Item: FromReflect + SSs,
{
    type Item = <Upstream::Item as Signal>::Item;

    fn register_signal(self, world: &mut World) -> SignalHandle {
        self.signal.register(world)
    }
}

/// Handle returned by [`SignalExt::register`] used for cleaning up the registered signal chain.
///
/// Contains the list of all system entities created by the specific `register` call
/// that produced this handle. Dropping the handle does *not* automatically clean up.
/// Use the [`cleanup`](SignalHandle::cleanup) method for explicit cleanup.
#[derive(Clone, Deref, DerefMut)]
pub struct SignalHandle(pub SignalSystem);

pub struct UpstreamIter<'w, 's, D: QueryData, F: QueryFilter>
where
    D::ReadOnly: WorldQuery<Item<'w> = &'w Upstream>,
{
    upstreams_query: &'w Query<'w, 's, D, F>,
    upstreams: VecDeque<SignalSystem>,
}

impl<'w, 's, D: QueryData, F: QueryFilter> UpstreamIter<'w, 's, D, F>
where
    D::ReadOnly: WorldQuery<Item<'w> = &'w Upstream>,
{
    /// Returns a new [`DescendantIter`].
    pub fn new(upstreams_query: &'w Query<'w, 's, D, F>, signal: SignalSystem) -> Self {
        UpstreamIter {
            upstreams_query,
            upstreams: upstreams_query
                .get(*signal)
                .into_iter()
                .flatten()
                .copied()
                .collect(),
        }
    }
}

impl<'w, 's, D: QueryData, F: QueryFilter> Iterator for UpstreamIter<'w, 's, D, F>
where
    D::ReadOnly: WorldQuery<Item<'w> = &'w Upstream>,
{
    type Item = SignalSystem;

    fn next(&mut self) -> Option<Self::Item> {
        let signal = self.upstreams.pop_front()?;

        if let Ok(upstream) = self.upstreams_query.get(*signal) {
            self.upstreams.extend(upstream);
        }

        Some(signal)
    }
}

pub struct DownstreamIter<'w, 's, D: QueryData, F: QueryFilter>
where
    D::ReadOnly: WorldQuery<Item<'w> = &'w Downstream>,
{
    downstreams_query: &'w Query<'w, 's, D, F>,
    downstreams: VecDeque<SignalSystem>,
}

impl<'w, 's, D: QueryData, F: QueryFilter> DownstreamIter<'w, 's, D, F>
where
    D::ReadOnly: WorldQuery<Item<'w> = &'w Downstream>,
{
    /// Returns a new [`DescendantIter`].
    pub fn new(downstreams_query: &'w Query<'w, 's, D, F>, signal: SignalSystem) -> Self {
        DownstreamIter {
            downstreams_query,
            downstreams: downstreams_query
                .get(*signal)
                .into_iter()
                .flatten()
                .copied()
                .collect(),
        }
    }
}

impl<'w, 's, D: QueryData, F: QueryFilter> Iterator for DownstreamIter<'w, 's, D, F>
where
    D::ReadOnly: WorldQuery<Item<'w> = &'w Downstream>,
{
    type Item = SignalSystem;

    fn next(&mut self) -> Option<Self::Item> {
        let signal = self.downstreams.pop_front()?;

        if let Ok(downstream) = self.downstreams_query.get(*signal) {
            self.downstreams.extend(downstream);
        }

        Some(signal)
    }
}

impl SignalHandle {
    /// Creates a new SignalHandle.
    /// This is crate-public to allow construction from other modules.
    pub(crate) fn new(signal: SignalSystem) -> Self {
        Self(signal)
    }

    /// decrements the ref count of this signal and all upstream signals, if the ref count reaches 0, despawns the signal and all its downstream
    pub fn cleanup(self, world: &mut World) {
        let signal = self.0;
        let _ = world.run_system_once(move |upstreams: Query<&Upstream>, downstreams: Query<&Downstream>, mut ref_counts: Query<&mut SignalReferenceCount>, mut commands: Commands| {
            for signal in [signal].into_iter().chain(UpstreamIter::new(&upstreams, signal)) {
                if let Ok(mut ref_count) = ref_counts.get_mut(*signal) {
                    if ref_count.decrement() == 0 {
                        if let Some(entity) = commands.get_entity(*signal) {
                            entity.despawn_recursive();
                        }
                        let mut downstream_bug = false;
                        for downstream in DownstreamIter::new(&downstreams, signal) {
                            downstream_bug = true;
                            if let Some(entity) = commands.get_entity(*downstream) {
                                entity.despawn_recursive();
                            }
                        }
                        if downstream_bug {
                            error!("downstreams should exist, this is a bug");
                        }
                    }
                }
            }
        });
    }
}

pub(crate) fn spawn_signal<I, O, IOO, IS, M>(world: &mut World, system: IS) -> SignalSystem
where
    I: FromReflect + SSs,
    O: FromReflect + SSs,
    IOO: Into<Option<O>> + SSs,
    IS: IntoSystem<In<I>, IOO, M> + SSs,
    M: SSs,
{
    let system = world.register_system(system);
    let entity = system.entity();
    world.entity_mut(entity).insert((
        SignalReferenceCount::new(),
        SystemRunner {
            runner: Arc::new(Box::new(move |world, input| {
                match I::from_reflect(input.as_ref()) {
                    Some(input) => match world.run_system_with_input(system, input) {
                        Ok(output) => {
                            if let Some(output) = Into::<Option<O>>::into(output) {
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
}

pub(crate) fn register_once_signal_from_system<I, O, IOO, IS, M>(system: IS) -> RegisterOnceSignal
where
    I: FromReflect + SSs,
    O: FromReflect + SSs,
    IOO: Into<Option<O>> + SSs,
    IS: IntoSystem<In<I>, IOO, M> + SSs,
    M: SSs,
{
    RegisterOnceSignal::System(Arc::new(Mutex::new(Some(Box::new(
        move |world: &mut World| spawn_signal(world, system)
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
        O: FromReflect + SSs,
        IOO: Into<Option<O>> + SSs,
        IS: IntoSystem<In<()>, IOO, M> + SSs,
        M: SSs,
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
        C: Component + FromReflect + GetTypeRegistration + Typed + Clone + SSs,
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
        R: Resource + FromReflect + GetTypeRegistration + Typed + Clone + SSs,
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
        Self::Item: FromReflect + SSs,
        O: FromReflect + SSs,
        IOO: Into<Option<O>> + SSs,
        IS: IntoSystem<In<Self::Item>, IOO, M>
            + Send
            + Sync
            + 'static,
        M: SSs;

    fn map_component<C>(self) -> MapComponent<Self, C>
    where
        Self: Sized,
        Self: Signal<Item = Entity>,
        C: Component + Clone + FromReflect + SSs;

    fn dedupe(self) -> Dedupe<Self>
    where
        Self: Sized,
        Self::Item: PartialEq + Clone + FromReflect + SSs;

    fn first(self) -> First<Self>
    where
        Self: Sized,
        Self::Item: FromReflect + SSs;

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
            FromReflect + GetTypeRegistration + Typed + SSs,
        Other::Item:
            FromReflect + GetTypeRegistration + Typed + SSs;
    
    fn flatten(self) -> Flatten<Self>
    where
        Self: Sized,
        Self::Item: FromReflect + Signal + Clone,
        <Self::Item as Signal>::Item: FromReflect + SSs;

    /// Adds a debug step to the signal chain that prints the value of the signal
    /// whenever it changes.
    fn debug(self) -> SignalDebug<Self>
    where
        Self: Sized,
        Self::Item: Debug + FromReflect + SSs;

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
        T::Item: FromReflect + SSs,
        O: FromReflect + SSs,
        IOO: Into<Option<O>> + SSs,
        IS: IntoSystem<In<T::Item>, IOO, M> + SSs,
        M: SSs,
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
        C: Component + Clone + FromReflect + SSs,
    {
        MapComponent {
            signal: self.map(|In(entity): In<Entity>, components: Query<&C>| components.get(entity).ok().cloned())
        }
    }

    fn dedupe(self) -> Dedupe<Self>
        where
            Self: Sized,
            Self::Item: PartialEq + Clone + FromReflect + SSs {
        Dedupe {
            signal: self.map(|In(current): In<Self::Item>, mut cache: Local<Option<Self::Item>>| {
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
            })
        }
    }

    fn first(self) -> First<Self>
    where
        Self: Sized,
        Self::Item: FromReflect + SSs,
    {
        First {
            signal: self.map(|In(item): In<Self::Item>, mut first: Local<bool>| {
                if *first {
                    None
                } else {
                    *first = true;
                    Some(item)
                }
            }),
        }
    }

    fn combine<Other>(self, other: Other) -> Combine<Self, Other>
    where
        Other: Signal,
        Self::Item:
            FromReflect + GetTypeRegistration + Typed + SSs,
        Other::Item:
            FromReflect + GetTypeRegistration + Typed + SSs,
    {
        let left_wrapper = self.map(|In(left): In<Self::Item>| {
            (Some(left), None::<Other::Item>)
        });
        let right_wrapper = other.map(|In(right): In<Other::Item>| {
            (None::<Self::Item>, Some(right))
        });
        let signal = register_once_signal_from_system::<_, (Self::Item, Other::Item), _, _, _>(
            move |In((left_option, right_option)): In<(
                Option<Self::Item>,
                Option<Other::Item>,
            )>,
                  mut left_cache: Local<Option<Self::Item>>,
                  mut right_cache: Local<Option<Other::Item>>| {
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
        Combine {
            left_wrapper,
            right_wrapper,
            signal,
        }
    }

    fn flatten(self) -> Flatten<Self>
    where
        Self: Sized,
        Self::Item: FromReflect + Signal + Clone,
        <Self::Item as Signal>::Item: FromReflect + SSs,
    {
        // TODO: forward with observer instead of mutex ?
        let cur = Arc::new(Mutex::new(None));
        Flatten { signal: self.map(move |In(signal): In<Self::Item>, world: &mut World, mut prev_system_option: Local<Option<(SignalSystem, SignalHandle)>>| {
            // TODO: is this registering/cleanup too expensive ?
            let signal_handle = signal.clone().register(world);
            let cur_system = signal_handle.0;
            signal_handle.cleanup(world);
            if prev_system_option.as_ref().is_some_and(|&(prev_system, _)| prev_system == cur_system).not() {
                if let Some((_, prev_forwarder)) = prev_system_option.take() {
                    prev_forwarder.cleanup(world);
                }
                let cur_clone = cur.clone();
                let forwarder = signal.map(move |In(item)| {
                    *cur_clone.lock().unwrap() = Some(item);
                });
                forwarder.register(world);
            }
            cur.lock().unwrap().take()
        }) }
    }

    fn debug(self) -> SignalDebug<Self>
    where
        Self: Sized,
        Self::Item: Debug + FromReflect + SSs,
    {
        let location = std::panic::Location::caller();
        SignalDebug {
            signal: self.map(move |In(item): In<Self::Item>| {
                debug!("[{}] {:#?}", location, item);
                item
            }),
        }
    }

    fn register(self, world: &mut World) -> SignalHandle {
        T::register_signal(self, world)
    }
}
