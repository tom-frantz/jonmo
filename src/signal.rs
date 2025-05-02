use crate::{process_signals_helper, tree::*, utils::*};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    prelude::*,
    query::{QueryData, QueryFilter, WorldQuery}, system::RunSystemOnce,
};
use bevy_hierarchy::prelude::*;
use bevy_log::prelude::*;
use bevy_reflect::{FromReflect, GetTypeRegistration, PartialReflect, Reflect, Typed};
use std::{
    collections::VecDeque,
    fmt::Debug,
    marker::PhantomData,
    sync::{Arc, LazyLock, Mutex},
};

pub(crate) enum RegisterOnceSignalInternal {
    // the function is just indirection required because IntoSystem isn't dyn compatible
    System(Option<Box<dyn FnOnce(&mut World) -> SignalSystem + Send + Sync + 'static>>),
    Registered(SignalSystem),
}

#[derive(Clone, Reflect)]
#[reflect(opaque)]
pub(crate) struct RegisterOnceSignal {
    inner: Arc<Mutex<RegisterOnceSignalInternal>>,
}

impl RegisterOnceSignal {
    /// Creates a new `RegisterOnceSignal` that will register the system only once.
    /// The system is provided as a closure that takes a mutable reference to the `World`.
    pub fn new<F: FnOnce(&mut World) -> SignalSystem + Send + Sync + 'static>(system: F) -> Self {
        RegisterOnceSignal {
            inner: Arc::new(Mutex::new(RegisterOnceSignalInternal::System(Some(
                Box::new(system),
            )))),
        }
    }
}

impl RegisterOnceSignal {
    /// Registers the system if it hasn't been registered yet.
    /// Returns the system ID of the registered system.
    pub fn register(&mut self, world: &mut World) -> SignalSystem {
        let mut guard = self.inner.lock().unwrap();
        match &mut *guard {
            RegisterOnceSignalInternal::System(f) => {
                let signal = f.take().unwrap()(world).into();
                *guard = RegisterOnceSignalInternal::Registered(signal);
                signal
            }
            RegisterOnceSignalInternal::Registered(signal) => {
                if let Ok(mut system) = world.get_entity_mut(**signal) {
                    if let Some(mut registration_count) =
                        system.get_mut::<SignalRegistrationCount>()
                    {
                        registration_count.increment();
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
#[reflect(opaque)]
pub struct Source<O>
where
    O: SSs + Clone,
{
    signal: RegisterOnceSignal,
    _marker: PhantomData<O>,
}

impl<O> Signal for Source<O>
where
    O: SSs + Clone,
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
    Upstream::Item: FromReflect + SSs,
    Switcher: Signal + FromReflect + SSs,
    Switcher::Item: FromReflect + SSs,
{
    pub(crate) signal: Flatten<Map<Upstream, Switcher>>,
}

impl<Upstream, Switcher> Signal for Switch<Upstream, Switcher>
where
    Upstream: Signal,
    Upstream::Item: FromReflect + SSs,
    Switcher: FromReflect + Signal,
    Switcher::Item: FromReflect + SSs,
{
    type Item = Switcher::Item;

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

fn signal_handle_cleanup_helper(
    world: &mut World,
    signals: impl IntoIterator<Item = SignalSystem>,
) {
    for signal in signals {
        if let Some(upstreams) = world.get::<Upstream>(*signal).cloned() {
            signal_handle_cleanup_helper(world, upstreams.0);
        }
        if let Ok(mut entity) = world.get_entity_mut(*signal) {
            if let Some(mut registration_count) = entity.get_mut::<SignalRegistrationCount>() {
                if registration_count.decrement() == 0 {
                    entity.remove::<Upstream>();
                    entity.remove::<Downstream>();
                }
            }
        }
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
        signal_handle_cleanup_helper(world, [self.0]);
    }
}

pub(crate) fn spawn_signal<I, O, IOO, F, M>(world: &mut World, system: F) -> SignalSystem
where
    I: FromReflect + SSs,
    O: FromReflect + SSs,
    IOO: Into<Option<O>> + SSs,
    F: IntoSystem<In<I>, IOO, M> + SSs,
    M: SSs,
{
    let system = world.register_system(system);
    let entity = system.entity();
    world.entity_mut(entity).insert((
        SignalRegistrationCount::new(),
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

// TODO: drop has to be impl for all signal structs, since they are the ones being cloned
pub(crate) static CLEANUP_SIGNALS: LazyLock<Mutex<Vec<SignalSystem>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));

impl Drop for RegisterOnceSignal {
    fn drop(&mut self) {
        if let RegisterOnceSignalInternal::Registered(signal) = &*self.inner.lock().unwrap() {
            CLEANUP_SIGNALS.lock().unwrap().push(*signal);
        }
    }
}

pub(crate) fn flush_cleanup_signals(world: &mut World) {
    let mut signals = CLEANUP_SIGNALS.lock().unwrap();
    for signal in signals.drain(..) {
        if let Ok(entity) = world.get_entity_mut(*signal) {
            // entity.try_despawn_recursive();
        }
    }
}

pub(crate) fn register_once_signal_from_system<I, O, IOO, F, M>(system: F) -> RegisterOnceSignal
where
    I: FromReflect + SSs,
    O: FromReflect + SSs,
    IOO: Into<Option<O>> + SSs,
    F: IntoSystem<In<I>, IOO, M> + SSs,
    M: SSs,
{
    RegisterOnceSignal::new(move |world: &mut World| spawn_signal(world, system))
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
    pub fn from_system<O, IOO, F, M>(system: F) -> Source<O>
    where
        O: FromReflect + SSs + Clone,
        IOO: Into<Option<O>> + SSs,
        F: IntoSystem<In<()>, IOO, M> + SSs,
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

    pub fn from_lazy_entity(entity: LazyEntity) -> Source<Entity> {
        Self::from_system(move |_: In<()>| entity.get())
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
    fn map<O, IOO, F, M>(self, system: F) -> Map<Self, O>
    where
        Self: Sized,
        Self::Item: FromReflect + SSs,
        O: FromReflect + SSs,
        IOO: Into<Option<O>> + SSs,
        F: IntoSystem<In<Self::Item>, IOO, M> + Send + Sync + 'static,
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

    fn has_component<C>(self) -> ContainsComponent<Self, C>
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

    fn filter<M>(
        self,
        predicate: impl IntoSystem<In<Self::Item>, bool, M> + Send + Sync + 'static,
    ) -> Filter<Self>
    where
        Self: Sized,
        Self::Item: Clone + FromReflect + SSs;

    fn filter_map<O, IOO, F, M>(self, system: F) -> FilterMap<Self, O>
    where
        Self: Sized,
        Self::Item: FromReflect + SSs,
        O: FromReflect + SSs,
        IOO: Into<Option<O>> + SSs,
        F: IntoSystem<In<Self::Item>, IOO, M> + Send + Sync + 'static,
        M: SSs;

    fn switch<S, F, M>(self, switcher: F) -> Switch<Self, S>
    where
        Self: Sized,
        Self::Item: FromReflect + SSs,
        S: Signal + Clone + FromReflect + SSs,
        S::Item: FromReflect + SSs + Clone,
        F: IntoSystem<In<Self::Item>, S, M> + Send + Sync + 'static,
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
    fn map<O, IOO, F, M>(self, system: F) -> Map<Self, O>
    where
        Self: Sized,
        Self::Item: FromReflect + SSs,
        O: FromReflect + SSs,
        IOO: Into<Option<O>> + SSs,
        F: IntoSystem<In<T::Item>, IOO, M> + SSs,
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

    fn has_component<C>(self) -> ContainsComponent<Self, C>
    where
        Self: Sized,
        Self: Signal<Item = Entity>,
        C: Component + Clone + FromReflect + SSs,
    {
        ContainsComponent {
            signal: self
                .map(|In(entity): In<Entity>, components: Query<&C>| components.contains(entity)),
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
                let forwarder = signal.map(clone!((cur) move |In(item)| {
                    *cur.lock().unwrap() = Some(item);
                }));
                let handle = forwarder.register(world);
                if let Ok(ancestor_orphans) = world.run_system_once(move |upstreams: Query<&Upstream>| {
                    UpstreamIter::new(&upstreams, handle.0).filter(|upstream| !upstreams.contains(**upstream)).collect::<Vec<_>>()
                }) {
                    process_signals_helper(world, ancestor_orphans, Box::new(()));
                }
                *prev_system_option = Some((cur_system, handle.clone()));
            }
            cur.lock().unwrap().take()
        }) }
    }

    fn eq(self, value: Self::Item) -> Eq<Self>
    where
        Self: Sized,
        Self::Item: PartialEq + FromReflect + SSs,
    {
        Eq {
            signal: self.map(move |In(item): In<Self::Item>| item == value),
        }
    }

    fn neq(self, value: Self::Item) -> Neq<Self>
    where
        Self: Sized,
        Self::Item: PartialEq + FromReflect + SSs,
    {
        Neq {
            signal: self.map(move |In(item): In<Self::Item>| item != value),
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

    fn filter<M>(
        self,
        predicate: impl IntoSystem<In<Self::Item>, bool, M> + Send + Sync + 'static,
    ) -> Filter<Self>
    where
        Self: Sized,
        Self::Item: Clone + FromReflect + SSs,
    {
        let signal = RegisterOnceSignal::new(move |world: &mut World| {
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
        });

        Filter {
            upstream: self,
            signal,
            _marker: PhantomData,
        }
    }

    fn filter_map<O, IOO, F, M>(self, system: F) -> FilterMap<Self, O>
    where
        Self: Sized,
        Self::Item: FromReflect + SSs,
        O: FromReflect + SSs,
        IOO: Into<Option<O>> + SSs,
        F: IntoSystem<In<Self::Item>, IOO, M> + Send + Sync + 'static,
        M: SSs,
    {
        FilterMap {
            upstream: self,
            signal: register_once_signal_from_system(system),
            _marker: PhantomData,
        }
    }

    fn switch<S, F, M>(self, switcher: F) -> Switch<Self, S>
    where
        Self: Sized,
        Self::Item: FromReflect + SSs,
        S: Signal + Clone + FromReflect + SSs,
        S::Item: FromReflect + SSs + Clone,
        F: IntoSystem<In<Self::Item>, S, M> + Send + Sync + 'static,
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

/// Struct representing a filter_map node in the signal chain definition. Implements [`Signal`].
#[derive(Clone)]
pub struct FilterMap<Upstream, O>
where
    Upstream: Signal,
    Upstream::Item: FromReflect + SSs,
    O: FromReflect + SSs,
{
    pub(crate) upstream: Upstream,
    pub(crate) signal: RegisterOnceSignal,
    _marker: PhantomData<O>,
}

impl<Upstream, O> Signal for FilterMap<Upstream, O>
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
        app.add_plugins((MinimalPlugins, JonmoPlugin));
        app.register_type::<TestData>();
        app
    }

    // Helper system to capture signal output
    fn capture_output<T: SSs + Clone + Debug>(
        In(value): In<T>,
        mut output: ResMut<SignalOutput<T>>,
    ) {
        output.0 = Some(value);
    }

    fn get_output<T: SSs + Clone + Debug>(world: &World) -> Option<T> {
        world.resource::<SignalOutput<T>>().0.clone()
    }

    #[test]
    fn test_map() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<i32>>();
        let signal = SignalBuilder::from_system(|_: In<()>| 1)
            .map(|In(x): In<i32>| x + 1)
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<i32>(app.world()), Some(2));
        signal.cleanup(app.world_mut());
    }

    #[test]
    fn test_component() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<TestData>>();
        let entity = app.world_mut().spawn(TestData(1)).id();
        let signal = SignalBuilder::from_entity(entity)
            .component::<TestData>()
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<TestData>(app.world()), Some(TestData(1)));
        signal.cleanup(app.world_mut());
    }

    #[test]
    fn test_component_option() {
        let mut app = create_test_app();
        app.insert_resource(SignalOutput::<Option<TestData>>::default());
        let entity_with = app.world_mut().spawn(TestData(1)).id();
        let entity_without = app.world_mut().spawn_empty().id();

        let signal = SignalBuilder::from_entity(entity_with)
            .component_option::<TestData>()
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(
            get_output::<Option<TestData>>(app.world()),
            Some(Some(TestData(1)))
        );
        signal.cleanup(app.world_mut());

        let signal = SignalBuilder::from_entity(entity_without)
            .component_option::<TestData>()
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<Option<TestData>>(app.world()), Some(None));
        signal.cleanup(app.world_mut());
    }

    #[test]
    fn test_has_component() {
        let mut app = create_test_app();
        app.insert_resource(SignalOutput::<bool>::default());
        let entity_with = app.world_mut().spawn(TestData(1)).id();
        let entity_without = app.world_mut().spawn_empty().id();

        let signal = SignalBuilder::from_entity(entity_with)
            .has_component::<TestData>()
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<bool>(app.world()), Some(true));
        signal.cleanup(app.world_mut());

        let signal = SignalBuilder::from_entity(entity_without)
            .has_component::<TestData>()
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<bool>(app.world()), Some(false));
        signal.cleanup(app.world_mut());
    }

    #[test]
    fn test_dedupe() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<i32>>();
        let counter = Arc::new(Mutex::new(0));

        let values = Arc::new(Mutex::new(vec![1, 1, 2, 3, 3, 3, 4]));
        let signal = SignalBuilder::from_system(clone!((values) move |_: In<()>| {
            let mut values_lock = values.lock().unwrap();
            if values_lock.is_empty() {
                None
            } else {
                Some(values_lock.remove(0))
            }
        }))
        .dedupe()
        .map(clone!((counter) move |In(val): In<i32>| {
            *counter.lock().unwrap() += 1;
            val
        }))
        .map(capture_output)
        .register(app.world_mut());

        for _ in 0..10 {
            app.update();
        }

        assert_eq!(get_output::<i32>(app.world()), Some(4));
        assert_eq!(*counter.lock().unwrap(), 4);
        assert_eq!(values.lock().unwrap().len(), 0);

        signal.cleanup(app.world_mut());
    }

    #[test]
    fn test_first() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<i32>>();
        let counter = Arc::new(Mutex::new(0));

        let values = Arc::new(Mutex::new(vec![1, 2, 3]));
        let signal = SignalBuilder::from_system(clone!((values) move |_: In<()>| {
            let mut values_lock = values.lock().unwrap();
            if values_lock.is_empty() {
                None
            } else {
                Some(values_lock.remove(0))
            }
        }))
        .first()
        .map(clone!((counter) move |In(val): In<i32>| {
            *counter.lock().unwrap() += 1;
            val
        }))
        .map(capture_output)
        .register(app.world_mut());

        app.update();
        app.update();
        app.update();

        assert_eq!(get_output::<i32>(app.world()), Some(1));
        assert_eq!(*counter.lock().unwrap(), 1);
        assert_eq!(values.lock().unwrap().len(), 0);

        signal.cleanup(app.world_mut());
    }

    #[test]
    fn test_combine() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<(i32, &'static str)>>();

        let signal = SignalBuilder::from_system(move |_: In<()>| 10)
            .combine(SignalBuilder::from_system(move |_: In<()>| "hello"))
            .map(capture_output)
            .register(app.world_mut());
        app.update();

        assert_eq!(
            get_output::<(i32, &'static str)>(app.world()),
            Some((10, "hello"))
        );
        signal.cleanup(app.world_mut());
    }

    #[test]
    fn test_flatten() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<i32>>();

        let signal_1 = SignalBuilder::from_system(|_: In<()>| 1);
        let signal_2 = SignalBuilder::from_system(|_: In<()>| 2);

        #[derive(Resource, Default)]
        struct SignalSelector(bool);
        app.init_resource::<SignalSelector>();

        let signal = SignalBuilder::from_system(move |_: In<()>, selector: Res<SignalSelector>| {
            if selector.0 {
                signal_1.clone()
            } else {
                signal_2.clone()
            }
        })
        .flatten()
        .map(capture_output)
        .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<i32>(app.world()), Some(2));

        app.world_mut().resource_mut::<SignalSelector>().0 = true;
        app.update();
        assert_eq!(get_output::<i32>(app.world()), Some(1));

        signal.cleanup(app.world_mut());
    }

    #[test]
    fn test_eq() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<bool>>();

        let source = SignalBuilder::from_system(|_: In<()>| 1);
        let signal = source
            .clone()
            .eq(1)
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<bool>(app.world()), Some(true));
        signal.cleanup(app.world_mut());

        let signal = source.eq(2).map(capture_output).register(app.world_mut());
        app.update();
        assert_eq!(get_output::<bool>(app.world()), Some(false));
        signal.cleanup(app.world_mut());
    }

    #[test]
    fn test_neq() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<bool>>();

        let source = SignalBuilder::from_system(|_: In<()>| 1);
        let signal = source
            .clone()
            .neq(2)
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<bool>(app.world()), Some(true));
        signal.cleanup(app.world_mut());

        let signal = source.neq(1).map(capture_output).register(app.world_mut());
        app.update();
        assert_eq!(get_output::<bool>(app.world()), Some(false));
        signal.cleanup(app.world_mut());
    }

    #[test]
    fn test_not() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<bool>>();

        let signal = SignalBuilder::from_system(|_: In<()>| true)
            .not()
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<bool>(app.world()), Some(false));
        signal.cleanup(app.world_mut());

        let signal = SignalBuilder::from_system(|_: In<()>| false)
            .not()
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<bool>(app.world()), Some(true));
        signal.cleanup(app.world_mut());
    }

    #[test]
    fn test_filter() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<i32>>();

        let values = Arc::new(Mutex::new(vec![1, 2, 3, 4, 5, 6]));
        let signal = SignalBuilder::from_system(move |_: In<()>| {
            let mut values_lock = values.lock().unwrap();
            if values_lock.is_empty() {
                None
            } else {
                Some(values_lock.remove(0))
            }
        })
        .filter(|In(x): In<i32>| x % 2 == 0)
        .map(capture_output)
        .register(app.world_mut());

        for _ in 0..10 {
            app.update();
        }

        assert_eq!(get_output::<i32>(app.world()), Some(6));

        signal.cleanup(app.world_mut());
    }

    #[test]
    fn test_filter_map() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<String>>();

        let values = Arc::new(Mutex::new(vec![1, 2, 3, 4, 5, 6]));
        let signal = SignalBuilder::from_system(move |_: In<()>| {
            let mut values_lock = values.lock().unwrap();
            if values_lock.is_empty() {
                None
            } else {
                Some(values_lock.remove(0))
            }
        })
        .filter_map(|In(x): In<i32>| {
            if x % 2 == 0 {
                Some(format!("even: {}", x))
            } else {
                None
            }
        })
        .map(capture_output::<String>) // Specify the type for capture_output
        .register(app.world_mut());

        for _ in 0..10 {
            app.update();
        }

        assert_eq!(get_output::<String>(app.world()), Some("even: 6".to_string()));

        signal.cleanup(app.world_mut());
    }

    #[test]
    fn test_switch() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<i32>>();

        let signal_1 = SignalBuilder::from_system(|_: In<()>| 1);
        let signal_2 = SignalBuilder::from_system(|_: In<()>| 2);

        #[derive(Resource, Default)]
        struct SwitcherToggle(bool);
        app.init_resource::<SwitcherToggle>();

        let signal =
            SignalBuilder::from_system(move |_: In<()>, mut toggle: ResMut<SwitcherToggle>| {
                let current = toggle.0;
                toggle.0 = !toggle.0;
                current
            })
            .switch(move |In(use_1): In<bool>| {
                if use_1 {
                    signal_1.clone()
                } else {
                    signal_2.clone()
                }
            })
            .map(capture_output)
            .register(app.world_mut());

        app.update();
        assert_eq!(get_output::<i32>(app.world()), Some(2));

        app.update();
        assert_eq!(get_output::<i32>(app.world()), Some(1));

        app.update();
        assert_eq!(get_output::<i32>(app.world()), Some(2));

        signal.cleanup(app.world_mut());
    }

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

    //     let handle = final_signal.register(app.world_mut());
    //     app.update();

    //     assert_eq!(get_output::<i32>(app.world()), Some(99));
    //     handle.cleanup(app.world_mut());
    // }

    // #[test]
    // fn test_register_and_cleanup() {
    //     let mut app = create_test_app();

    //     let source = SignalBuilder::from_system(|_: In<()>| Some(1));
    //     let mapped = source.map(|In(x): In<i32>| x + 1);

    //     // Register
    //     let handle = mapped.clone().register(app.world_mut());
    //     // Need a way to reliably get the entities. Let's assume the source system is created first.
    //     // This is fragile. A better way might involve querying by a marker component.
    //     let mut query = app.world.query::<(Entity, &SignalReferenceCount)>();
    //     let entities: Vec<Entity> = query.iter(app.world()).map(|(e, _)| e).collect();
    //     assert!(entities.len() >= 2, "Expected at least source and map systems");
    //     // Assuming source is the first one registered by the builder, and map is the one in the handle
    //     let source_entity = entities[0]; // Highly dependent on registration order
    //     let map_entity = handle.0.entity();

    //     assert!(app.world.get::<SignalReferenceCount>(source_entity).is_some());
    //     assert!(app.world.get::<SignalReferenceCount>(map_entity).is_some());
    //     assert_eq!(app.world.get::<SignalReferenceCount>(source_entity).unwrap().count(), 1);
    //     assert_eq!(app.world.get::<SignalReferenceCount>(map_entity).unwrap().count(), 1);

    //     // Register again (should increment ref count)
    //     let handle2 = mapped.clone().register(app.world_mut());
    //     assert_eq!(app.world.get::<SignalReferenceCount>(source_entity).unwrap().count(), 2);
    //     // The map entity might be different if register_once_signal_from_system re-registers.
    //     // Let's check the handle's entity specifically.
    //     let map_entity2 = handle2.0.entity();
    //     assert_eq!(map_entity, map_entity2, "Expected same map entity on re-register");
    //     assert_eq!(app.world.get::<SignalReferenceCount>(map_entity).unwrap().count(), 2);

    //     // Cleanup first handle
    //     handle.cleanup(app.world_mut());
    //     app.update(); // Allow cleanup system to run
    //     assert_eq!(app.world.get::<SignalReferenceCount>(source_entity).unwrap().count(), 1);
    //     assert_eq!(app.world.get::<SignalReferenceCount>(map_entity).unwrap().count(), 1);

    //     // Cleanup second handle (should despawn)
    //     handle2.cleanup(app.world_mut());
    //     app.update(); // Allow cleanup system to run

    //     // Entities should be despawned
    //     assert!(app.world.get_entity(source_entity).is_none());
    //     assert!(app.world.get_entity(map_entity).is_none());
    // }
}
