use crate::{tree::*, utils::SSs};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    prelude::*,
    query::{QueryData, QueryFilter, WorldQuery},
    system::RunSystemOnce,
};
use bevy_hierarchy::prelude::*;
use bevy_log::prelude::*;
use bevy_reflect::{FromReflect, GetTypeRegistration, PartialReflect, Reflect, Typed};
use std::{
    collections::VecDeque,
    fmt::Debug,
    marker::PhantomData,
    sync::{Arc, Mutex},
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
    O: SSs,
{
    type Item = O;

    fn register_signal(mut self, world: &mut World) -> SignalHandle {
        SignalHandle::new(self.signal.register(world))
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
        let signal = self.signal.register(world);
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

pub struct ComponentOption<Upstream, C>
where
    Upstream: Signal<Item = Entity>,
    C: Component + Clone + FromReflect + GetTypeRegistration + Typed + SSs,
{
    pub(crate) signal: Map<Upstream, Option<C>>,
}

impl<Upstream, C> Signal for ComponentOption<Upstream, C>
where
    Upstream: Signal<Item = Entity>,
    C: Component + Clone + FromReflect + GetTypeRegistration + Typed + SSs,
{
    type Item = Option<C>;

    fn register_signal(self, world: &mut World) -> SignalHandle {
        self.signal.register(world)
    }
}

pub struct ContainsComponent<Upstream, C>
where
    Upstream: Signal<Item = Entity>,
    Upstream::Item: SSs,
    C: Component + Clone + SSs,
{
    pub(crate) signal: Map<Upstream, bool>,
    _marker: PhantomData<C>,
}

impl<Upstream, C> Signal for ContainsComponent<Upstream, C>
where
    Upstream: Signal<Item = Entity>,
    Upstream::Item: FromReflect + SSs,
    C: Component + Clone + FromReflect + SSs,
{
    type Item = bool;

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
        let signal = self.signal.register(world);
        pipe_signal(world, left_upstream, signal);
        pipe_signal(world, right_upstream, signal);
        SignalHandle::new(signal)
    }
}

pub struct Flatten<Upstream>
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

pub struct Eq<Upstream>
where
    Upstream: Signal,
    Upstream::Item: PartialEq + FromReflect + SSs,
{
    pub(crate) signal: Map<Upstream, bool>,
}

impl<Upstream> Signal for Eq<Upstream>
where
    Upstream: Signal,
    Upstream::Item: PartialEq + FromReflect + SSs,
{
    type Item = bool;

    fn register_signal(self, world: &mut World) -> SignalHandle {
        self.signal.register(world)
    }
}

pub struct Neq<Upstream>
where
    Upstream: Signal,
    Upstream::Item: PartialEq + FromReflect + SSs,
{
    pub(crate) signal: Map<Upstream, bool>,
}

impl<Upstream> Signal for Neq<Upstream>
where
    Upstream: Signal,
    Upstream::Item: PartialEq + FromReflect + SSs,
{
    type Item = bool;

    fn register_signal(self, world: &mut World) -> SignalHandle {
        self.signal.register(world)
    }
}

pub struct Not<Upstream>
where
    Upstream: Signal,
    <Upstream as Signal>::Item: std::ops::Not + FromReflect + SSs,
    <<Upstream as Signal>::Item as std::ops::Not>::Output: FromReflect + SSs,
{
    pub(crate) signal: Map<Upstream, <Upstream::Item as std::ops::Not>::Output>,
}

impl<Upstream> Signal for Not<Upstream>
where
    Upstream: Signal,
    <Upstream as Signal>::Item: std::ops::Not + FromReflect + SSs,
    <<Upstream as Signal>::Item as std::ops::Not>::Output: FromReflect + SSs,
{
    type Item = <Upstream::Item as std::ops::Not>::Output;

    fn register_signal(self, world: &mut World) -> SignalHandle {
        self.signal.register(world)
    }
}

pub struct Filter<Upstream>
where
    Upstream: Signal,
    Upstream::Item: Clone + FromReflect + SSs,
{
    pub(crate) upstream: Upstream,
    pub(crate) signal: RegisterOnceSignal,
    _marker: PhantomData<bool>,
}

impl<Upstream> Signal for Filter<Upstream>
where
    Upstream: Signal,
    Upstream::Item: Clone + FromReflect + SSs,
{
    type Item = Upstream::Item;

    fn register_signal(mut self, world: &mut World) -> SignalHandle {
        let SignalHandle(upstream) = self.upstream.register(world);
        let signal = self.signal.register(world);
        pipe_signal(world, upstream, signal);
        SignalHandle::new(signal)
    }
}

pub struct Switch<Upstream, Switcher>
where
    Upstream: Signal,
    Upstream::Item: Signal + FromReflect + SSs,
    Switcher: Signal + FromReflect + SSs,
    Switcher::Item: FromReflect + SSs,
{
    pub(crate) signal: Flatten<Map<Upstream, Switcher>>,
}

impl<Upstream, Switcher> Signal for Switch<Upstream, Switcher>
where
    Upstream: Signal,
    Upstream::Item: Signal + FromReflect + SSs,
    Switcher: FromReflect + Signal,
    Switcher::Item: FromReflect + SSs,
{
    type Item = Upstream::Item;

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
        let _ = world.run_system_once(
            move |upstreams: Query<&Upstream>,
                  downstreams: Query<&Downstream>,
                  mut ref_counts: Query<&mut SignalReferenceCount>,
                  mut commands: Commands| {
                for signal in [signal]
                    .into_iter()
                    .chain(UpstreamIter::new(&upstreams, signal))
                {
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
            },
        );
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
        move |world: &mut World| spawn_signal(world, system),
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

    /// Creates a signal chain that starts by observing changes to a specific component `C`
    /// on a given `entity`.
    ///
    /// The signal emits the new value of the component `C` whenever it changes on the entity.
    /// Requires the component `C` to implement `Component`, `FromReflect`, `Clone`, `Send`, `Sync`, and `'static`.
    /// Internally uses [`SignalBuilder::from_system`].
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
        IS: IntoSystem<In<Self::Item>, IOO, M> + Send + Sync + 'static,
        M: SSs;

    fn component<C>(self) -> MapComponent<Self, C>
    where
        Self: Sized,
        Self: Signal<Item = Entity>,
        C: Component + Clone + FromReflect + SSs;

    fn component_option<C>(self) -> ComponentOption<Self, C>
    where
        Self: Sized,
        Self: Signal<Item = Entity>,
        C: Component + Clone + FromReflect + GetTypeRegistration + Typed + SSs;

    fn contains_component<C>(self) -> ContainsComponent<Self, C>
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
        Self::Item: FromReflect + GetTypeRegistration + Typed + SSs,
        Other::Item: FromReflect + GetTypeRegistration + Typed + SSs;

    fn flatten(self) -> Flatten<Self>
    where
        Self: Sized,
        Self::Item: FromReflect + Signal + Clone,
        <Self::Item as Signal>::Item: FromReflect + SSs;

    fn eq(self, value: Self::Item) -> Eq<Self>
    where
        Self: Sized,
        Self::Item: PartialEq + FromReflect + SSs;

    fn neq(self, value: Self::Item) -> Neq<Self>
        where
            Self: Sized,
            Self::Item: PartialEq + FromReflect + SSs;

    fn not(self) -> Not<Self>
    where
        Self: Sized,
        <Self as Signal>::Item: std::ops::Not + FromReflect + SSs,
        <<Self as Signal>::Item as std::ops::Not>::Output: FromReflect + SSs;

    fn filter<M>(self, predicate: impl IntoSystem<In<Self::Item>, bool, M> + Send + Sync + 'static) -> Filter<Self>
    where
        Self: Sized,
        Self::Item: Clone + FromReflect + SSs;

    fn switch<S, IS, M>(self, switcher: IS) -> Switch<Self, S>
    where
        Self: Sized,
        Self::Item: Signal + FromReflect + SSs,
        S: Signal + Clone + FromReflect + SSs,
        S::Item: FromReflect + SSs,
        IS: IntoSystem<In<Self::Item>, S, M> + Send + Sync + 'static,
        M: SSs;

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
        Self: Sized,
        Self::Item: FromReflect + SSs,
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

    fn component<C>(self) -> MapComponent<Self, C>
    where
        Self: Sized,
        Self: Signal<Item = Entity>,
        C: Component + Clone + FromReflect + SSs,
    {
        MapComponent {
            signal: self.map(|In(entity): In<Entity>, components: Query<&C>| {
                components.get(entity).ok().cloned()
            }),
        }
    }

    fn component_option<C>(self) -> ComponentOption<Self, C>
    where
        Self: Sized,
        Self: Signal<Item = Entity>,
        C: Component + Clone + FromReflect + GetTypeRegistration + Typed + SSs,
    {
        ComponentOption {
            signal: self.map(|In(entity): In<Entity>, components: Query<&C>| {
                Some(components.get(entity).ok().cloned())
            }),
        }
    }

    fn contains_component<C>(self) -> ContainsComponent<Self, C>
    where
        Self: Sized,
        Self: Signal<Item = Entity>,
        C: Component + Clone + FromReflect + SSs,
    {
        ContainsComponent {
            signal: self.map(|In(entity): In<Entity>, components: Query<&C>| {
                components.contains(entity)
            }),
            _marker: PhantomData,
        }
    }

    fn dedupe(self) -> Dedupe<Self>
    where
        Self: Sized,
        Self::Item: PartialEq + Clone + FromReflect + SSs,
    {
        Dedupe {
            signal: self.map(
                |In(current): In<Self::Item>, mut cache: Local<Option<Self::Item>>| {
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
                },
            ),
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
        Self::Item: FromReflect + GetTypeRegistration + Typed + SSs,
        Other::Item: FromReflect + GetTypeRegistration + Typed + SSs,
    {
        let left_wrapper = self.map(|In(left): In<Self::Item>| (Some(left), None::<Other::Item>));
        let right_wrapper =
            other.map(|In(right): In<Other::Item>| (None::<Self::Item>, Some(right)));
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
            if !prev_system_option.as_ref().is_some_and(|&(prev_system, _)| prev_system == cur_system) {
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

    fn eq(self, value: Self::Item) -> Eq<Self>
    where
        Self: Sized,
        Self::Item: PartialEq + FromReflect + SSs
    {
        Eq {
            signal: self.map(move |In(item): In<Self::Item>| {
                item == value
            })
        }
    }

    fn neq(self, value: Self::Item) -> Neq<Self>
    where
        Self: Sized,
        Self::Item: PartialEq + FromReflect + SSs,
    {
        Neq {
            signal: self.map(move |In(item): In<Self::Item>| {
                item != value
            }),
        }
    }

    fn not(self) -> Not<Self>
    where
        Self: Sized,
        <Self as Signal>::Item: std::ops::Not + FromReflect + SSs,
        <<Self as Signal>::Item as std::ops::Not>::Output: FromReflect + SSs,
    {
        Not {
            signal: self.map(|In(item): In<Self::Item>| std::ops::Not::not(item)),
        }
    }

    fn filter<M>(self, predicate: impl IntoSystem<In<Self::Item>, bool, M> + Send + Sync + 'static) -> Filter<Self>
    where
        Self: Sized,
        Self::Item: Clone + FromReflect + SSs,

    {
        let signal = RegisterOnceSignal::System(Arc::new(Mutex::new(Some(Box::new(
            move |world: &mut World| {
                let system = world.register_system(predicate);
                let wrapper_system = move |In(item): In<Self::Item>, world: &mut World| {
                    match world.run_system_with_input(system, item.clone()) {
                        Ok(true) => Some(item),
                        Ok(false) | Err(_) => None, // terminate on false or error
                    }
                };
                let signal = register_signal::<_, Self::Item, _, _, _>(world, wrapper_system);
                // just attach the system to the lifetime of the signal
                world.entity_mut(*signal).add_child(system.entity());
                signal.into()
            },
        )))));

        Filter {
            upstream: self,
            signal,
            _marker: PhantomData,
        }
    }

    fn switch<S, IS, M>(self, switcher: IS) -> Switch<Self, S>
    where
        Self: Sized,
        Self::Item: Signal + FromReflect + SSs,
        S: Signal + Clone + FromReflect + SSs,
        S::Item: FromReflect + SSs,
        IS: IntoSystem<In<Self::Item>, S, M> + Send + Sync + 'static,
        M: SSs,
    {
        Switch {
            signal: self.map(switcher).flatten(),
        }
    }

    /// Adds a debug step to the signal chain that prints the value of the signal
    /// whenever it changes.
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

#[cfg(test)]
mod tests {
    use crate::JonmoPlugin;

    use super::*;
    // Import Bevy prelude for MinimalPlugins and other common items
    use bevy::prelude::*;
    use std::sync::{Arc, Mutex};

    // Helper component and resource for testing
    #[derive(Component, Clone, Debug, PartialEq, Reflect, Default)] // Add Default
    struct TestData(i32);

    #[derive(Resource, Default, Debug)]
    struct SignalOutput<T: SSs + Clone + Debug>(Option<T>);

    fn create_test_app() -> App {
        let mut app = App::new();
        // Use Bevy's MinimalPlugins
        app.add_plugins((MinimalPlugins, JonmoPlugin));
        // Register reflected types
        app.register_type::<TestData>();
        app
    }

    // Helper system to capture signal output
    fn capture_output<T: SSs + Clone + Debug>(In(value): In<T>, mut output: ResMut<SignalOutput<T>>) {
        output.0 = Some(value);
    }

    fn get_output<T: SSs + Clone + Debug>(world: &World) -> Option<T> {
        world.resource::<SignalOutput<T>>().0.clone()
    }

    #[test]
    fn test_map() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<i32>>();

        let source = SignalBuilder::from_system(|_: In<()>| 1);
        let mapped = source.map(|In(x): In<i32>| x + 1);
        let final_signal = mapped.map(capture_output::<i32>);

        let handle = final_signal.register(app.world_mut());
        app.update();

        assert_eq!(get_output::<i32>(app.world_mut()), Some(2));
        handle.cleanup(app.world_mut());
    }

    #[test]
    fn test_component() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<TestData>>();

        let entity = app.world_mut().spawn(TestData(42)).id();
        let source = SignalBuilder::from_entity(entity);
        let component_signal = source.component::<TestData>();
        let final_signal = component_signal.map(capture_output::<TestData>);

        let handle = final_signal.register(app.world_mut());
        app.update();

        assert_eq!(get_output::<TestData>(app.world()), Some(TestData(42)));
        handle.cleanup(app.world_mut());
    }

    #[test]
    fn test_component_option() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<Option<TestData>>>();

        let entity_with = app.world_mut().spawn(TestData(42)).id();
        let entity_without = app.world_mut().spawn_empty().id();

        // Test with component
        let source_with = SignalBuilder::from_entity(entity_with);
        let component_opt_with = source_with.component_option::<TestData>();
        let final_signal_with = component_opt_with.map(capture_output::<Option<TestData>>);
        let handle_with = final_signal_with.register(app.world_mut());
        app.update();
        assert_eq!(get_output::<Option<TestData>>(app.world()), Some(Some(TestData(42))));
        handle_with.cleanup(app.world_mut());

        // Reset output resource for the next part of the test
        app.world_mut().resource_mut::<SignalOutput<Option<TestData>>>().0 = None;

        // Test without component
        let source_without = SignalBuilder::from_entity(entity_without);
        let component_opt_without = source_without.component_option::<TestData>();
        let final_signal_without = component_opt_without.map(capture_output::<Option<TestData>>);
        let handle_without = final_signal_without.register(app.world_mut());
        app.update();
        assert_eq!(get_output::<Option<TestData>>(app.world()), Some(None));
        handle_without.cleanup(app.world_mut());
    }

    #[test]
    fn test_contains_component() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<bool>>();

        let entity_with = app.world_mut().spawn(TestData(42)).id();
        let entity_without = app.world_mut().spawn_empty().id();

        // Test with component
        let source_with = SignalBuilder::from_entity(entity_with);
        let contains_with = source_with.contains_component::<TestData>();
        let final_signal_with = contains_with.map(capture_output::<bool>);
        let handle_with = final_signal_with.register(app.world_mut());
        app.update();
        assert_eq!(get_output::<bool>(app.world()), Some(true));
        handle_with.cleanup(app.world_mut());

        // Reset output resource
        app.world_mut().resource_mut::<SignalOutput<bool>>().0 = None;

        // Test without component
        let source_without = SignalBuilder::from_entity(entity_without);
        let contains_without = source_without.contains_component::<TestData>();
        let final_signal_without = contains_without.map(capture_output::<bool>);
        let handle_without = final_signal_without.register(app.world_mut());
        app.update();
        assert_eq!(get_output::<bool>(app.world()), Some(false));
        handle_without.cleanup(app.world_mut());
    }

    // #[test]
    // fn test_dedupe() {
    //     let mut app = create_test_app();
    //     app.init_resource::<SignalOutput<i32>>();
    //     let output_res = app.world.resource::<SignalOutput<i32>>().clone();
    //     let counter = Arc::new(Mutex::new(0));

    //     let values = Arc::new(Mutex::new(vec![1, 1, 2, 3, 3, 3, 4]));
    //     let source = SignalBuilder::from_system(move |_: In<()>| {
    //         let mut values_lock = values.lock().unwrap();
    //         if values_lock.is_empty() {
    //             None
    //         } else {
    //             Some(values_lock.remove(0))
    //         }
    //     });

    //     let deduped = source.dedupe();
    //     let final_signal = deduped.map(move |In(val): In<i32>| {
    //         *counter.lock().unwrap() += 1;
    //         capture_output(output_res.clone())(In(val));
    //         Some(()) // Ensure map system returns Option
    //     });

    //     let handle = final_signal.register(&mut app.world);

    //     // Run multiple updates to process the vector
    //     for _ in 0..10 {
    //         app.update();
    //     }

    //     assert_eq!(get_output::<i32>(&app.world), Some(4)); // Last unique value
    //     assert_eq!(*counter.lock().unwrap(), 4); // Should have triggered 4 times (1, 2, 3, 4)

    //     handle.cleanup(&mut app.world);
    // }

    // #[test]
    // fn test_first() {
    //     let mut app = create_test_app();
    //     app.init_resource::<SignalOutput<i32>>();
    //     let output_res = app.world.resource::<SignalOutput<i32>>().clone();
    //     let counter = Arc::new(Mutex::new(0));

    //     let values = Arc::new(Mutex::new(vec![1, 2, 3]));
    //     let source = SignalBuilder::from_system(move |_: In<()>| {
    //         let mut values_lock = values.lock().unwrap();
    //         if values_lock.is_empty() {
    //             None
    //         } else {
    //             Some(values_lock.remove(0))
    //         }
    //     });

    //     let first_signal = source.first();
    //     let final_signal = first_signal.map(move |In(val): In<i32>| {
    //         *counter.lock().unwrap() += 1;
    //         capture_output(output_res.clone())(In(val));
    //         Some(()) // Ensure map system returns Option
    //     });

    //     let handle = final_signal.register(&mut app.world);

    //     app.update(); // Process 1
    //     app.update(); // Process 2 (should be ignored by first)
    //     app.update(); // Process 3 (should be ignored by first)

    //     assert_eq!(get_output::<i32>(&app.world), Some(1)); // Only the first value
    //     assert_eq!(*counter.lock().unwrap(), 1); // Should have triggered only once

    //     handle.cleanup(&mut app.world);
    // }

    // #[test]
    // fn test_combine() {
    //     let mut app = create_test_app();
    //     app.init_resource::<SignalOutput<(i32, String)>>();
    //     let output_res = app.world.resource::<SignalOutput<(i32, String)>>().clone();

    //     let source1_val = Arc::new(Mutex::new(Some(10)));
    //     let source1 = SignalBuilder::from_system(move |_: In<()>| source1_val.lock().unwrap().take());

    //     let source2_val = Arc::new(Mutex::new(Some("hello".to_string())));
    //     let source2 = SignalBuilder::from_system(move |_: In<()>| source2_val.lock().unwrap().take());

    //     let combined = source1.combine(source2);
    //     let final_signal = combined.map(capture_output(output_res));

    //     let handle = final_signal.register(&mut app.world);
    //     app.update();

    //     assert_eq!(get_output::<(i32, String)>(&app.world), Some((10, "hello".to_string())));
    //     handle.cleanup(&mut app.world);
    // }

    // // Flatten test needs adjustment for how signals are created and returned.
    // // Let's use a resource to hold the inner signal definition.
    // #[derive(Resource, Clone)]
    // struct InnerSignalDef(Source<i32>);

    // #[test]
    // fn test_flatten() {
    //     let mut app = create_test_app();
    //     app.init_resource::<SignalOutput<i32>>();
    //     let output_res = app.world.resource::<SignalOutput<i32>>().clone();

    //     // Define two potential inner signals
    //     let inner_signal_100 = SignalBuilder::from_system(|_: In<()>| Some(100));
    //     let inner_signal_200 = SignalBuilder::from_system(|_: In<()>| Some(200));

    //     // Resource to control which inner signal is active
    //     #[derive(Resource, Default)] struct InnerSelector(bool);
    //     app.insert_resource::<InnerSignalDef>(inner_signal_100.clone()); // Start with 100

    //     // Outer signal emits the *definition* of the inner signal based on the selector
    //     let outer_signal = SignalBuilder::from_system(move |_: In<()>, selector: Res<InnerSelector>| {
    //          if selector.0 {
    //              Some(inner_signal_100.clone()) // Clone the definition
    //          } else {
    //              Some(inner_signal_200.clone())
    //          }
    //     });

    //     let flattened = outer_signal.flatten();
    //     let final_signal = flattened.map(capture_output(output_res));

    //     let handle = final_signal.register(&mut app.world);

    //     app.update(); // Outer runs (selector=false), emits inner_200 def, flatten registers inner_200, inner_200 runs
    //     assert_eq!(get_output::<i32>(&app.world), Some(200));

    //     // Change the selector and update again
    //     app.world.resource_mut::<InnerSelector>().0 = true;
    //     app.update(); // Outer runs (selector=true), emits inner_100 def, flatten cleans up old, registers inner_100, inner_100 runs
    //     assert_eq!(get_output::<i32>(&app.world), Some(100));

    //     handle.cleanup(&mut app.world);
    // }

    // #[test]
    // fn test_eq() {
    //     let mut app = create_test_app();
    //     app.init_resource::<SignalOutput<bool>>();
    //     let output_res = app.world.resource::<SignalOutput<bool>>().clone();

    //     let source = SignalBuilder::from_system(|_: In<()>| Some(5));
    //     let eq_signal = source.eq(5);
    //     let final_signal = eq_signal.map(capture_output(output_res.clone()));
    //     let handle = final_signal.register(&mut app.world);
    //     app.update();
    //     assert_eq!(get_output::<bool>(&app.world), Some(true));
    //     handle.cleanup(&mut app.world);

    //     *output_res.0.lock().unwrap() = None; // Reset
    //     let source_neq = SignalBuilder::from_system(|_: In<()>| Some(6));
    //     let eq_signal_neq = source_neq.eq(5);
    //     let final_signal_neq = eq_signal_neq.map(capture_output(output_res));
    //     let handle_neq = final_signal_neq.register(&mut app.world);
    //     app.update();
    //     assert_eq!(get_output::<bool>(&app.world), Some(false));
    //     handle_neq.cleanup(&mut app.world);
    // }

    // #[test]
    // fn test_neq() {
    //     let mut app = create_test_app();
    //     app.init_resource::<SignalOutput<bool>>();
    //     let output_res = app.world.resource::<SignalOutput<bool>>().clone();

    //     let source = SignalBuilder::from_system(|_: In<()>| Some(5));
    //     let neq_signal = source.neq(6);
    //     let final_signal = neq_signal.map(capture_output(output_res.clone()));
    //     let handle = final_signal.register(&mut app.world);
    //     app.update();
    //     assert_eq!(get_output::<bool>(&app.world), Some(true));
    //     handle.cleanup(&mut app.world);

    //     *output_res.0.lock().unwrap() = None; // Reset
    //     let source_eq = SignalBuilder::from_system(|_: In<()>| Some(5));
    //     let neq_signal_eq = source_eq.neq(5);
    //     let final_signal_eq = neq_signal_eq.map(capture_output(output_res));
    //     let handle_eq = final_signal_eq.register(&mut app.world);
    //     app.update();
    //     assert_eq!(get_output::<bool>(&app.world), Some(false));
    //     handle_eq.cleanup(&mut app.world);
    // }

    // #[test]
    // fn test_not() {
    //     let mut app = create_test_app();
    //     app.init_resource::<SignalOutput<bool>>();
    //     let output_res = app.world.resource::<SignalOutput<bool>>().clone();

    //     let source_true = SignalBuilder::from_system(|_: In<()>| Some(true));
    //     let not_true = source_true.not();
    //     let final_signal_true = not_true.map(capture_output(output_res.clone()));
    //     let handle_true = final_signal_true.register(&mut app.world);
    //     app.update();
    //     assert_eq!(get_output::<bool>(&app.world), Some(false));
    //     handle_true.cleanup(&mut app.world);

    //     *output_res.0.lock().unwrap() = None; // Reset
    //     let source_false = SignalBuilder::from_system(|_: In<()>| Some(false));
    //     let not_false = source_false.not();
    //     let final_signal_false = not_false.map(capture_output(output_res));
    //     let handle_false = final_signal_false.register(&mut app.world);
    //     app.update();
    //     assert_eq!(get_output::<bool>(&app.world), Some(true));
    //     handle_false.cleanup(&mut app.world);
    // }

    // #[test]
    // fn test_filter() {
    //     let mut app = create_test_app();
    //     app.init_resource::<SignalOutput<i32>>();
    //     let output_res = app.world.resource::<SignalOutput<i32>>().clone();
    //     let counter = Arc::new(Mutex::new(0));

    //     let values = Arc::new(Mutex::new(vec![1, 2, 3, 4, 5, 6]));
    //     let source = SignalBuilder::from_system(move |_: In<()>| {
    //         let mut values_lock = values.lock().unwrap();
    //         if values_lock.is_empty() {
    //             None
    //         } else {
    //             Some(values_lock.remove(0))
    //         }
    //     });

    //     let filtered = source.filter(|In(x): In<i32>| x % 2 == 0); // Keep even numbers
    //     let final_signal = filtered.map(move |In(val): In<i32>| {
    //         *counter.lock().unwrap() += 1;
    //         capture_output(output_res.clone())(In(val));
    //         Some(()) // Ensure map system returns Option
    //     });

    //     let handle = final_signal.register(&mut app.world);

    //     for _ in 0..10 {
    //         app.update();
    //     }

    //     assert_eq!(get_output::<i32>(&app.world), Some(6)); // Last even number
    //     assert_eq!(*counter.lock().unwrap(), 3); // Should have triggered 3 times (2, 4, 6)

    //     handle.cleanup(&mut app.world);
    // }

    // #[test]
    // fn test_switch() {
    //     let mut app = create_test_app();
    //     app.init_resource::<SignalOutput<i32>>();
    //     let output_res = app.world.resource::<SignalOutput<i32>>().clone();

    //     // Define the two potential inner signals
    //     let signal_a = SignalBuilder::from_system(|_: In<()>| Some(10));
    //     let signal_b = SignalBuilder::from_system(|_: In<()>| Some(20));

    //     #[derive(Resource, Default)] struct SwitcherToggle(bool);
    //     app.init_resource::<SwitcherToggle>();

    //     let switcher_signal = SignalBuilder::from_system(move |_: In<()>, mut toggle: ResMut<SwitcherToggle>| {
    //         let current = toggle.0;
    //         toggle.0 = !toggle.0;
    //         Some(current) // Emit false, then true, then false...
    //     });

    //     // The switcher function maps the boolean from switcher_signal to one of the inner signals
    //     let switched = switcher_signal.switch(move |In(use_a): In<bool>| {
    //         if use_a {
    //             signal_a.clone() // Clone the definition
    //         } else {
    //             signal_b.clone()
    //         }
    //     });

    //     let final_signal = switched.map(capture_output(output_res));
    //     let handle = final_signal.register(&mut app.world);

    //     app.update(); // Switcher=false, inner=signal_b (20)
    //     assert_eq!(get_output::<i32>(&app.world), Some(20));

    //     app.update(); // Switcher=true, inner=signal_a (10)
    //     assert_eq!(get_output::<i32>(&app.world), Some(10));

    //     app.update(); // Switcher=false, inner=signal_b (20)
    //     assert_eq!(get_output::<i32>(&app.world), Some(20));

    //     handle.cleanup(&mut app.world);
    // }

    // #[test]
    // fn test_debug() {
    //     // Test that it compiles and runs without panicking.
    //     // Actual debug output verification is not practical here.
    //     let mut app = create_test_app();
    //     app.init_resource::<SignalOutput<i32>>();
    //     let output_res = app.world.resource::<SignalOutput<i32>>().clone();

    //     let source = SignalBuilder::from_system(|_: In<()>| Some(99));
    //     let debugged = source.debug();
    //     let final_signal = debugged.map(capture_output(output_res));

    //     let handle = final_signal.register(&mut app.world);
    //     app.update();

    //     assert_eq!(get_output::<i32>(&app.world), Some(99));
    //     handle.cleanup(&mut app.world);
    // }

    // #[test]
    // fn test_register_and_cleanup() {
    //     let mut app = create_test_app();

    //     let source = SignalBuilder::from_system(|_: In<()>| Some(1));
    //     let mapped = source.map(|In(x): In<i32>| x + 1);

    //     // Register
    //     let handle = mapped.clone().register(&mut app.world);
    //     // Need a way to reliably get the entities. Let's assume the source system is created first.
    //     // This is fragile. A better way might involve querying by a marker component.
    //     let mut query = app.world.query::<(Entity, &SignalReferenceCount)>();
    //     let entities: Vec<Entity> = query.iter(&app.world).map(|(e, _)| e).collect();
    //     assert!(entities.len() >= 2, "Expected at least source and map systems");
    //     // Assuming source is the first one registered by the builder, and map is the one in the handle
    //     let source_entity = entities[0]; // Highly dependent on registration order
    //     let map_entity = handle.0.entity();

    //     assert!(app.world.get::<SignalReferenceCount>(source_entity).is_some());
    //     assert!(app.world.get::<SignalReferenceCount>(map_entity).is_some());
    //     assert_eq!(app.world.get::<SignalReferenceCount>(source_entity).unwrap().count(), 1);
    //     assert_eq!(app.world.get::<SignalReferenceCount>(map_entity).unwrap().count(), 1);

    //     // Register again (should increment ref count)
    //     let handle2 = mapped.clone().register(&mut app.world);
    //     assert_eq!(app.world.get::<SignalReferenceCount>(source_entity).unwrap().count(), 2);
    //     // The map entity might be different if register_once_signal_from_system re-registers.
    //     // Let's check the handle's entity specifically.
    //     let map_entity2 = handle2.0.entity();
    //     assert_eq!(map_entity, map_entity2, "Expected same map entity on re-register");
    //     assert_eq!(app.world.get::<SignalReferenceCount>(map_entity).unwrap().count(), 2);


    //     // Cleanup first handle
    //     handle.cleanup(&mut app.world);
    //     app.update(); // Allow cleanup system to run
    //     assert_eq!(app.world.get::<SignalReferenceCount>(source_entity).unwrap().count(), 1);
    //     assert_eq!(app.world.get::<SignalReferenceCount>(map_entity).unwrap().count(), 1);

    //     // Cleanup second handle (should despawn)
    //     handle2.cleanup(&mut app.world);
    //     app.update(); // Allow cleanup system to run

    //     // Entities should be despawned
    //     assert!(app.world.get_entity(source_entity).is_none());
    //     assert!(app.world.get_entity(map_entity).is_none());
    // }
}


