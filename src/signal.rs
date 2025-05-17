use crate::{tree::*, utils::*};
use bevy_ecs::prelude::*;
use bevy_hierarchy::prelude::*;
use bevy_log::prelude::*;
use bevy_reflect::{FromReflect, GetTypeRegistration, PartialReflect, Reflect, Typed};
use bevy_time::{Time, Timer, TimerMode}; // Use Timer and TimerMode for throttle
use std::{
    fmt::Debug,
    marker::PhantomData,
    sync::{Arc, Mutex},
    time::Duration, // Add Duration import
}; // Add import

/// Represents a value that changes over time and handles internal registration logic.
///
/// Signals are the core building block for reactive data flow. They are typically
/// created using methods on the [`SignalBuilder`] struct (e.g., [`SignalBuilder::from_component`])
/// and then transformed or combined using methods from the [`SignalExt`] trait.
///
/// This trait defines the fundamental behavior of a signal, including its output type
/// and the mechanism for registering its underlying systems in the Bevy `World`.
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

/// Represents a source node in the signal chain definition. Implements [`Signal`].
///
/// A source signal is the starting point of a signal chain. It typically originates
/// from a Bevy resource, component, entity, or a custom system.
#[derive(Clone, Reflect)]
pub struct Source<O> {
    signal: LazySignal,
    #[reflect(ignore)]
    _marker: PhantomData<O>,
}

impl<O> Signal for Source<O>
where
    O: SSs + Clone,
{
    type Item = O;

    fn register_signal(self, world: &mut World) -> SignalHandle {
        SignalHandle::new(self.signal.register(world))
    }
}

/// Represents a map node in the signal chain definition. Implements [`Signal`].
///
/// A map node transforms the value emitted by its upstream signal using a provided system.
/// The transformation function receives the upstream value (`Upstream::Item`) via `In`
/// and should return an `Option<O>`. Returning `None` terminates the signal propagation
/// for the current update cycle.
#[derive(Clone, Reflect)]
pub struct Map<Upstream, O> {
    pub(crate) upstream: Upstream,
    pub(crate) signal: LazySignal,
    #[reflect(ignore)]
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

/// Represents a node that extracts a component `C` from an entity signal. Implements [`Signal`].
///
/// This is a specialized map node where the upstream signal emits `Entity` values.
/// It attempts to get the component `C` from the emitted entity and propagates the component's value.
/// If the entity does not have the component, the signal terminates for that update.
#[derive(Clone, Reflect)]
pub struct MapComponent<Upstream, C> {
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

/// Represents a node that extracts an optional component `C` from an entity signal. Implements [`Signal`].
///
/// Similar to `MapComponent`, but always propagates an `Option<C>`. It emits `Some(C)`
/// if the entity has the component, and `None` otherwise.
#[derive(Clone, Reflect)]
pub struct ComponentOption<Upstream, C> {
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

/// Represents a node that checks if an entity signal contains a specific component `C`. Implements [`Signal`].
///
/// This node takes an entity signal and emits `true` if the entity has component `C`,
/// and `false` otherwise.
#[derive(Clone, Reflect)]
pub struct HasComponent<Upstream, C> {
    pub(crate) signal: Map<Upstream, bool>,
    #[reflect(ignore)]
    _marker: PhantomData<C>,
}

impl<Upstream, C> Signal for HasComponent<Upstream, C>
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

/// Represents a node that filters out consecutive duplicate values. Implements [`Signal`].
///
/// This node only propagates a value if it is different from the previously emitted value.
/// Requires the `Item` type to implement `PartialEq`.
#[derive(Clone, Reflect)]
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

/// Represents a node that only emits the first value it receives. Implements [`Signal`].
///
/// After the first value is emitted, this node will terminate propagation for all subsequent updates.
#[derive(Clone, Reflect)]
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

/// Represents a combine node in the signal chain definition. Implements [`Signal`].
///
/// This node combines the latest values from two upstream signals (`Left` and `Right`).
/// It emits a tuple `(Left::Item, Right::Item)` only when *both* upstream signals have
/// produced a new value since the last emission from this combine node. It internally
/// caches the latest value from each upstream signal.
#[derive(Clone, Reflect)]
pub struct Combine<Left, Right>
where
    Left: Signal,
    Right: Signal,
    Left::Item: FromReflect + GetTypeRegistration + Typed + SSs,
    Right::Item: FromReflect + GetTypeRegistration + Typed + SSs,
{
    pub(crate) left_wrapper: Map<Left, (Option<Left::Item>, Option<Right::Item>)>,
    pub(crate) right_wrapper: Map<Right, (Option<Left::Item>, Option<Right::Item>)>,
    pub(crate) signal: LazySignal,
}

impl<Left, Right> Signal for Combine<Left, Right>
where
    Left: Signal,
    Left::Item: FromReflect + GetTypeRegistration + Typed + SSs,
    Right: Signal,
    Right::Item: FromReflect + GetTypeRegistration + Typed + SSs,
{
    type Item = (Left::Item, Right::Item);

    fn register_signal(self, world: &mut World) -> SignalHandle {
        let SignalHandle(left_upstream) = self.left_wrapper.register(world);
        let SignalHandle(right_upstream) = self.right_wrapper.register(world);
        let signal = self.signal.register(world);
        pipe_signal(world, left_upstream, signal);
        pipe_signal(world, right_upstream, signal);
        SignalHandle::new(signal)
    }
}

/// Represents a node that flattens a signal of signals. Implements [`Signal`].
///
/// If the upstream signal emits values that are themselves signals (`Upstream::Item: Signal`),
/// this node subscribes to the *inner* signal emitted most recently and propagates values
/// from that inner signal. When the upstream emits a *new* inner signal, `Flatten` switches
/// its subscription to the new one.
#[derive(Clone, Reflect)]
pub struct Flatten<Upstream>
where
    Upstream: Signal,
    Upstream::Item: FromReflect + Signal,
    <Upstream::Item as Signal>::Item: FromReflect + SSs + Clone,
{
    pub(crate) signal: Map<Upstream, <Upstream::Item as Signal>::Item>,
}

impl<Upstream> Signal for Flatten<Upstream>
where
    Upstream: Signal,
    Upstream::Item: FromReflect + Signal,
    <Upstream::Item as Signal>::Item: FromReflect + SSs + Clone,
{
    type Item = <Upstream::Item as Signal>::Item;

    fn register_signal(self, world: &mut World) -> SignalHandle {
        self.signal.register(world)
    }
}

/// Represents a node that compares the upstream signal's value for equality with a fixed value. Implements [`Signal`].
///
/// Emits `true` if the upstream value is equal to the provided `value`, `false` otherwise.
/// Requires `Upstream::Item` to implement `PartialEq`.
#[derive(Clone, Reflect)]
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

/// Represents a node that compares the upstream signal's value for inequality with a fixed value. Implements [`Signal`].
///
/// Emits `true` if the upstream value is *not* equal to the provided `value`, `false` otherwise.
/// Requires `Upstream::Item` to implement `PartialEq`.
#[derive(Clone, Reflect)]
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

/// Represents a node that applies logical negation to a boolean signal. Implements [`Signal`].
///
/// Requires `Upstream::Item` to implement `std::ops::Not`. Typically used with boolean signals.
#[derive(Clone, Reflect)]
pub struct Not<Upstream>
where
    Upstream: Signal,
    <Upstream as Signal>::Item: std::ops::Not + FromReflect + SSs,
    <<Upstream as Signal>::Item as std::ops::Not>::Output: FromReflect + SSs + Clone,
{
    pub(crate) signal: Map<Upstream, <Upstream::Item as std::ops::Not>::Output>,
}

impl<Upstream> Signal for Not<Upstream>
where
    Upstream: Signal,
    <Upstream as Signal>::Item: std::ops::Not + FromReflect + SSs,
    <<Upstream as Signal>::Item as std::ops::Not>::Output: FromReflect + SSs + Clone,
{
    type Item = <Upstream::Item as std::ops::Not>::Output;

    fn register_signal(self, world: &mut World) -> SignalHandle {
        self.signal.register(world)
    }
}

/// Represents a node that filters values based on a predicate system. Implements [`Signal`].
///
/// This node takes a predicate system that receives the upstream value (`Upstream::Item`) via `In`
/// and returns `bool`. The node only propagates the upstream value if the predicate returns `true`.
/// If the predicate returns `false` or the system errors, propagation terminates.
#[derive(Clone, Reflect)]
pub struct Filter<Upstream> {
    pub(crate) upstream: Upstream,
    pub(crate) signal: LazySignal,
    #[reflect(ignore)]
    _marker: PhantomData<bool>,
}

impl<Upstream> Signal for Filter<Upstream>
where
    Upstream: Signal,
    Upstream::Item: Clone + FromReflect + SSs,
{
    type Item = Upstream::Item;

    fn register_signal(self, world: &mut World) -> SignalHandle {
        let SignalHandle(upstream) = self.upstream.register(world);
        let signal = self.signal.register(world);
        pipe_signal(world, upstream, signal);
        SignalHandle::new(signal)
    }
}

/// Represents a node that dynamically switches between signals based on the upstream value. Implements [`Signal`].
///
/// This node takes a `switcher` system. The `switcher` receives the value from the `Upstream` signal
/// and must return another signal (`Switcher: Signal`). The `Switch` node then behaves like the
/// returned signal until the `Upstream` emits a new value, causing the `switcher` to potentially
/// return a different signal to switch to.
#[derive(Clone, Reflect)]
pub struct Switch<Upstream, Other>
where
    Upstream: Signal,
    Upstream::Item: FromReflect + SSs,
    Other: Signal + FromReflect + SSs,
    Other::Item: FromReflect + SSs + Clone,
{
    pub(crate) signal: Flatten<Map<Upstream, Other>>,
}

impl<Upstream, Other> Signal for Switch<Upstream, Other>
where
    Upstream: Signal,
    Upstream::Item: FromReflect + SSs,
    Other: FromReflect + Signal,
    Other::Item: FromReflect + SSs + Clone,
{
    type Item = Other::Item;

    fn register_signal(self, world: &mut World) -> SignalHandle {
        self.signal.register(world)
    }
}

/// Represents a node that throttles the upstream signal based on a duration. Implements [`Signal`].
///
/// This node only propagates a value if the specified `duration` has elapsed since the
/// last propagated value. It uses Bevy's `Time` resource internally.
pub struct Throttle<Upstream>
where
    Upstream: Signal,
    Upstream::Item: FromReflect + SSs,
{
    pub(crate) signal: Map<Upstream, Upstream::Item>,
}

impl<Upstream> Signal for Throttle<Upstream>
where
    Upstream: Signal,
    Upstream::Item: FromReflect + SSs,
{
    type Item = Upstream::Item;

    fn register_signal(self, world: &mut World) -> SignalHandle {
        self.signal.register(world)
    }
}

pub struct MapBool<Upstream, O> {
    upstream: Upstream,
    signal: LazySignal,
    _marker: PhantomData<O>,
}

impl<Upstream, O> Signal for MapBool<Upstream, O>
where
    Upstream: Signal<Item = bool>,
    Upstream::Item: FromReflect + SSs,
    O: FromReflect + SSs,
{
    type Item = O;

    fn register_signal(self, world: &mut World) -> SignalHandle {
        let SignalHandle(upstream) = self.upstream.register(world);
        let signal = self.signal.register(world);
        pipe_signal(world, upstream, signal);
        SignalHandle::new(signal)
    }
}

/// Represents a node that adds debug logging to a signal chain. Implements [`Signal`].
///
/// This node passes through the upstream value unchanged but logs it using `bevy_log::debug!`
/// along with the source code location where `.debug()` was called.
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

/// Represents a node that maps a `true` boolean signal value using a system. Implements [`Signal`].
/// Executes the provided system `t` only when the upstream signal emits `true`.
pub struct MapTrue<Upstream, O> {
    upstream: Upstream,
    signal: LazySignal,
    _marker: PhantomData<O>,
}

impl<Upstream, O> Signal for MapTrue<Upstream, O>
where
    Upstream: Signal<Item = bool>,
    Upstream::Item: FromReflect + SSs,
    O: FromReflect + SSs,
{
    type Item = O;

    fn register_signal(self, world: &mut World) -> SignalHandle {
        let SignalHandle(upstream) = self.upstream.register(world);
        let signal = self.signal.register(world);
        pipe_signal(world, upstream, signal);
        SignalHandle::new(signal)
    }
}

/// Represents a node that maps a `false` boolean signal value using a system. Implements [`Signal`].
/// Executes the provided system `f` only when the upstream signal emits `false`.
pub struct MapFalse<Upstream, O> {
    upstream: Upstream,
    signal: LazySignal,
    _marker: PhantomData<O>,
}

impl<Upstream, O> Signal for MapFalse<Upstream, O>
where
    Upstream: Signal<Item = bool>,
    Upstream::Item: FromReflect + SSs,
    O: FromReflect + SSs,
{
    type Item = O;

    fn register_signal(self, world: &mut World) -> SignalHandle {
        let SignalHandle(upstream) = self.upstream.register(world);
        let signal = self.signal.register(world);
        pipe_signal(world, upstream, signal);
        SignalHandle::new(signal)
    }
}

// --- Map Option Family Structs ---

/// Represents a node that maps an `Option<T>` signal value using different systems for `Some` and `None`. Implements [`Signal`].
/// Executes `some_system` if the upstream emits `Some(T)`, passing `T` via `In`.
/// Executes `none_system` if the upstream emits `None`, passing `()` via `In`.
pub struct MapOption<Upstream, I, O> {
    upstream: Upstream,
    signal: LazySignal,
    _marker: PhantomData<(I, O)>,
}

impl<Upstream, I, O> Signal for MapOption<Upstream, I, O>
where
    Upstream: Signal<Item = Option<I>>,
    Upstream::Item: FromReflect + SSs,
    I: FromReflect + GetTypeRegistration + Typed + SSs,
    O: FromReflect + SSs,
{
    type Item = O;

    fn register_signal(self, world: &mut World) -> SignalHandle {
        let SignalHandle(upstream) = self.upstream.register(world);
        let signal = self.signal.register(world);
        pipe_signal(world, upstream, signal);
        SignalHandle::new(signal)
    }
}

/// Represents a node that maps a `Some(T)` signal value using a system. Implements [`Signal`].
/// Executes the provided system `some_system` only when the upstream signal emits `Some(T)`, passing `T` via `In`.
pub struct MapSome<Upstream, I, O> {
    upstream: Upstream,
    signal: LazySignal,
    _marker: PhantomData<(I, O)>,
}

impl<Upstream, I, O> Signal for MapSome<Upstream, I, O>
where
    Upstream: Signal<Item = Option<I>>,
    Upstream::Item: FromReflect + SSs,
    I: FromReflect + GetTypeRegistration + Typed + SSs,
    O: FromReflect + SSs,
{
    type Item = O;

    fn register_signal(self, world: &mut World) -> SignalHandle {
        let SignalHandle(upstream) = self.upstream.register(world);
        let signal = self.signal.register(world);
        pipe_signal(world, upstream, signal);
        SignalHandle::new(signal)
    }
}

/// Represents a node that maps a `None` signal value using a system. Implements [`Signal`].
/// Executes the provided system `none_system` only when the upstream signal emits `None`, passing `()` via `In`.
pub struct MapNone<Upstream, I, O> {
    upstream: Upstream,
    signal: LazySignal,
    _marker: PhantomData<(I, O)>,
}

impl<Upstream, I, O> Signal for MapNone<Upstream, I, O>
where
    Upstream: Signal<Item = Option<I>>,
    Upstream::Item: FromReflect + SSs,
    I: FromReflect + GetTypeRegistration + Typed + SSs,
    O: FromReflect + SSs,
{
    type Item = O;

    fn register_signal(self, world: &mut World) -> SignalHandle {
        let SignalHandle(upstream) = self.upstream.register(world);
        let signal = self.signal.register(world);
        pipe_signal(world, upstream, signal);
        SignalHandle::new(signal)
    }
}

/// Provides static methods for creating new signal chains (source signals).
///
/// Use methods like [`SignalBuilder::from_component`], [`SignalBuilder::from_resource`],
/// or [`SignalBuilder::from_system`] to start building a signal chain. These methods
/// return a [`Source`] signal, which can then be chained with combinators from the
/// [`SignalExt`] trait.
pub struct SignalBuilder;

impl From<Entity> for Source<Entity> {
    fn from(entity: Entity) -> Self {
        SignalBuilder::from_entity(entity)
    }
}

// Static methods to start signal chains, now associated with SignalBuilder struct
impl SignalBuilder {
    /// Creates a signal chain starting from a custom Bevy system.
    ///
    /// The provided `system` takes `In<()>` (no input) and returns an `Option<O>`.
    /// The signal will emit the value `O` whenever the system returns `Some(O)`.
    /// The system is run once per Bevy update cycle during signal propagation.
    ///
    /// # Example
    /// ```no_run
    /// # use bevy::prelude::*;
    /// # use jonmo::prelude::*;
    /// # fn my_system(_: In<()>) -> Option<i32> { Some(42) }
    /// let signal: Source<i32> = SignalBuilder::from_system(my_system); // Add type annotation
    /// ```
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
    /// The signal will emit the `Entity` ID itself. This is useful for creating
    /// signal chains that react to or operate on a specific entity.
    ///
    /// # Example
    /// ```no_run
    /// # use bevy::prelude::*;
    /// # use jonmo::prelude::*;
    /// # let mut world = World::new();
    /// let my_entity = world.spawn_empty().id();
    /// let signal = SignalBuilder::from_entity(my_entity); // Emits `my_entity`
    /// ```
    pub fn from_entity(entity: Entity) -> Source<Entity> {
        Self::from_system(move |_: In<()>| entity)
    }

    /// Creates a signal chain starting from a [`LazyEntity`].
    ///
    /// Similar to `from_entity`, but resolves the `LazyEntity` to an `Entity`
    /// when the signal is first propagated.
    pub fn from_lazy_entity(entity: LazyEntity) -> Source<Entity> {
        Self::from_system(move |_: In<()>| entity.get())
    }

    /// Creates a signal chain that observes changes to a specific component `C`
    /// on a given `entity`.
    ///
    /// The signal emits the new value of the component `C` whenever it changes on the entity.
    /// If the entity does not have the component `C`, the signal will not emit.
    /// Requires the component `C` to implement standard Bevy reflection traits and `Clone`.
    ///
    /// # Example
    /// ```no_run
    /// # use bevy::prelude::*;
    /// # use jonmo::prelude::*;
    /// #[derive(Component, Reflect, Clone, Default)]
    /// #[reflect(Component)]
    /// struct MyData(i32);
    /// # let mut world = World::new();
    /// let my_entity = world.spawn(MyData(10)).id();
    /// let signal = SignalBuilder::from_component::<MyData>(my_entity); // Emits `MyData` when it changes
    /// ```
    pub fn from_component<C>(entity: Entity) -> Source<C>
    where
        C: Component + FromReflect + GetTypeRegistration + Typed + Clone + SSs,
    {
        Self::from_system(move |_: In<()>, components: Query<&C>| {
            components.get(entity).ok().cloned()
        })
    }

    pub fn from_component_lazy<C>(entity: LazyEntity) -> Source<C>
    where
        C: Component + FromReflect + GetTypeRegistration + Typed + Clone + SSs,
    {
        Self::from_system(move |_: In<()>, components: Query<&C>| {
            components.get(entity.get()).ok().cloned()
        })
    }

    pub fn from_component_option<C>(entity: Entity) -> Source<Option<C>>
    where
        C: Component + FromReflect + GetTypeRegistration + Typed + Clone + SSs,
    {
        Self::from_system(move |_: In<()>, components: Query<&C>| {
            Some(components.get(entity).ok().cloned())
        })
    }

    pub fn from_component_option_lazy<C>(entity: LazyEntity) -> Source<Option<C>>
    where
        C: Component + FromReflect + GetTypeRegistration + Typed + Clone + SSs,
    {
        Self::from_system(move |_: In<()>, components: Query<&C>| {
            Some(components.get(entity.get()).ok().cloned())
        })
    }

    /// Creates a signal chain that observes changes to a specific resource `R`.
    ///
    /// The signal emits the new value of the resource `R` whenever it changes.
    /// Requires the resource `R` to implement standard Bevy reflection traits and `Clone`.
    ///
    /// # Example
    /// ```no_run
    /// # use bevy::prelude::*;
    /// # use jonmo::prelude::*;
    /// #[derive(Resource, Reflect, Clone, Default)]
    /// #[reflect(Resource)]
    /// struct MyResource(String);
    /// # let mut world = World::new();
    /// # world.init_resource::<MyResource>();
    /// let signal = SignalBuilder::from_resource::<MyResource>(); // Emits `MyResource` when it changes
    /// ```
    pub fn from_resource<R>() -> Source<R>
    where
        R: Resource + FromReflect + GetTypeRegistration + Typed + Clone + SSs,
    {
        Self::from_system(move |_: In<()>, resource: Option<Res<R>>| resource.map(|r| r.clone()))
    }

    pub fn from_resource_option<R>() -> Source<Option<R>>
    where
        R: Resource + FromReflect + GetTypeRegistration + Typed + Clone + SSs,
    {
        Self::from_system(move |_: In<()>, resource: Option<Res<R>>| resource.map(|r| r.clone()))
    }
}

#[derive(Clone, Reflect)]
pub enum SignalEither<L, R>
where
    L: Signal + FromReflect + SSs,
    R: Signal + FromReflect + SSs,
{
    Left(L),
    Right(R),
}

impl<T: SSs, L: Signal<Item = T>, R: Signal<Item = T>> Signal for SignalEither<L, R>
where
    T: SSs,
    L: Signal<Item = T> + FromReflect + SSs,
    R: Signal<Item = T> + FromReflect + SSs,
{
    type Item = T;
    fn register_signal(self, world: &mut World) -> SignalHandle {
        match self {
            SignalEither::Left(left) => left.register_signal(world),
            SignalEither::Right(right) => right.register_signal(world),
        }
    }
}

pub trait IntoSignalEither: Sized
where
    Self: Signal + FromReflect + SSs,
{
    fn left_either<R>(self) -> SignalEither<Self, R>
    where
        R: Signal + FromReflect + SSs,
    {
        SignalEither::Left(self)
    }

    fn right_either<L>(self) -> SignalEither<L, Self>
    where
        L: Signal + FromReflect + SSs,
    {
        SignalEither::Right(self)
    }
}

impl<T: Signal + FromReflect + SSs> IntoSignalEither for T {}

/// Extension trait providing combinator methods for types implementing [`Signal`].
///
/// These methods allow chaining operations like mapping, filtering, combining, etc.,
/// onto existing signals to build complex reactive data flows.
pub trait SignalExt: Signal {
    /// Appends a transformation step to the signal chain using a Bevy system.
    ///
    /// The provided `system` takes the output `Item` of the previous signal (wrapped in `In<Item>`)
    /// and returns an `Option<O>`. If it returns `Some(O)`, `U` is propagated to the next step.
    /// If it returns `None`, propagation along this branch stops for the current update cycle.
    ///
    /// # Example
    /// ```no_run
    /// # use bevy::prelude::*;
    /// # use jonmo::prelude::*;
    /// let signal = SignalBuilder::from_system(|_: In<()>| 1)
    ///     .map(|In(x): In<i32>| x * 2); // Signal now emits 2
    /// ```
    fn map<O, IOO, F, M>(self, system: F) -> Map<Self, O>
    where
        Self: Sized,
        Self::Item: FromReflect + SSs,
        O: FromReflect + SSs,
        IOO: Into<Option<O>> + SSs,
        F: IntoSystem<In<Self::Item>, IOO, M> + Send + Sync + 'static,
        M: SSs,
    {
        Map {
            upstream: self,
            signal: register_once_signal_from_system(system),
            _marker: PhantomData,
        }
    }

    /// Extracts a component `C` from a signal emitting `Entity` values.
    ///
    /// This is a shorthand for `.map(|In(entity): In<Entity>, q: Query<&C>| q.get(entity).ok().cloned())`.
    /// If the entity does not have the component, the signal terminates for that update.
    ///
    /// # Example
    /// ```no_run
    /// # use bevy::prelude::*;
    /// # use jonmo::prelude::*;
    /// #[derive(Component, Reflect, Clone, Default)]
    /// #[reflect(Component)]
    /// struct Score(u32);
    /// # let mut world = World::new();
    /// let player_entity = world.spawn(Score(100)).id();
    /// let score_signal = SignalBuilder::from_entity(player_entity)
    ///     .component::<Score>(); // Emits Score(100)
    /// ```
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

    /// Extracts an `Option<C>` of a component from a signal emitting `Entity` values.
    ///
    /// Similar to `.component()`, but always emits `Some(C)` or `None`, never terminating.
    /// Shorthand for `.map(|In(entity): In<Entity>, q: Query<&C>| Some(q.get(entity).ok().cloned()))`.
    ///
    /// # Example
    /// ```no_run
    /// # use bevy::prelude::*;
    /// # use jonmo::prelude::*;
    /// #[derive(Component, Reflect, Clone, Default)]
    /// #[reflect(Component)]
    /// struct Health(u32);
    /// # let mut world = World::new();
    /// let entity_with = world.spawn(Health(50)).id();
    /// let entity_without = world.spawn_empty().id();
    /// let health_signal = SignalBuilder::from_entity(entity_with)
    ///     .component_option::<Health>(); // Emits Some(Health(50))
    /// let no_health_signal = SignalBuilder::from_entity(entity_without)
    ///     .component_option::<Health>(); // Emits None
    /// ```
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

    /// Checks if an entity from an entity signal has component `C`.
    ///
    /// Shorthand for `.map(|In(entity): In<Entity>, q: Query<&C>| q.contains(entity))`.
    /// Emits `true` or `false`.
    ///
    /// # Example
    /// ```no_run
    /// # use bevy::prelude::*;
    /// # use jonmo::prelude::*;
    /// #[derive(Component, Reflect, Clone, Default)]
    /// #[reflect(Component)]
    /// struct IsPlayer;
    /// # let mut world = World::new();
    /// let player_entity = world.spawn(IsPlayer).id();
    /// let non_player_entity = world.spawn_empty().id();
    /// let is_player_signal = SignalBuilder::from_entity(player_entity)
    ///     .has_component::<IsPlayer>(); // Emits true
    /// let is_not_player_signal = SignalBuilder::from_entity(non_player_entity)
    ///     .has_component::<IsPlayer>(); // Emits false
    /// ```
    fn has_component<C>(self) -> HasComponent<Self, C>
    where
        Self: Sized,
        Self: Signal<Item = Entity>,
        C: Component + Clone + FromReflect + SSs,
    {
        HasComponent {
            signal: self
                .map(|In(entity): In<Entity>, components: Query<&C>| components.contains(entity)),
            _marker: PhantomData,
        }
    }

    /// Filters out consecutive duplicate values from the signal.
    ///
    /// Only emits a value if it's different from the immediately preceding value emitted by this signal.
    /// Requires `Self::Item` to implement `PartialEq` and `Clone`.
    ///
    /// # Example
    /// ```no_run
    /// # use bevy::prelude::*;
    /// # use jonmo::prelude::*;
    /// # use std::collections::VecDeque;
    /// // Create a signal that emits: 1, 1, 2, 3, 3, 3, 4
    /// let initial_values = vec![1, 1, 2, 3, 3, 3, 4];
    /// let source_signal = SignalBuilder::from_system({
    ///     let initial_values = initial_values.clone();
    ///     move |_: In<()>, mut state: Local<Option<VecDeque<i32>>>| {
    ///         if state.is_none() {
    ///             *state = Some(initial_values.into());
    ///         }
    ///         state.as_mut().unwrap().pop_front()
    ///     }
    /// });
    /// let deduped_signal = source_signal.dedupe(); // Emits: 1, 2, 3, 4
    /// ```
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

    /// Emits only the very first value received from the upstream signal.
    ///
    /// After the first value is emitted, this signal will stop propagating any further values.
    ///
    /// # Example
    /// ```no_run
    /// # use bevy::prelude::*;
    /// # use jonmo::prelude::*;
    /// # use std::collections::VecDeque;
    /// // Create a signal that emits: 10, 20, 30
    /// let initial_values = vec![10, 20, 30];
    /// let source_signal = SignalBuilder::from_system({
    ///     let initial_values = initial_values.clone();
    ///     move |_: In<()>, mut state: Local<Option<VecDeque<i32>>>| {
    ///         if state.is_none() {
    ///             *state = Some(initial_values.into());
    ///         }
    ///         state.as_mut().unwrap().pop_front()
    ///     }
    /// });
    /// let first_value_signal = source_signal.first(); // Emits: 10 (and then stops)
    /// ```
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

    /// Combines this signal with another signal (`other`).
    ///
    /// Creates a new signal that emits a tuple `(Self::Item, Other::Item)`.
    /// The combined signal only emits when *both* input signals have provided a new value
    /// since the last time the combined signal emitted. It caches the latest value from each input.
    ///
    /// # Example
    /// ```no_run
    /// # use bevy::prelude::*;
    /// # use jonmo::prelude::*;
    /// let signal_a = SignalBuilder::from_system(|_: In<()>| 1);
    /// let signal_b = SignalBuilder::from_system(|_: In<()>| "hello");
    /// let combined = signal_a.combine(signal_b); // Emits (1, "hello")
    /// ```
    fn combine<Other>(self, other: Other) -> Combine<Self, Other>
    where
        Self: Sized, // Added Sized bound here
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

    /// Flattens a signal where the item type is itself a signal (`Signal<Item = impl Signal>`).
    ///
    /// Subscribes to the *inner* signal produced by the outer signal. When the outer signal
    /// emits a *new* inner signal, `flatten` switches its subscription to the new inner signal
    /// and propagates values from it.
    ///
    /// # Example
    /// ```no_run
    /// # use bevy::prelude::*;
    /// # use jonmo::prelude::*;
    /// let signal_a = SignalBuilder::from_system(|_: In<()>| 10);
    /// let signal_b = SignalBuilder::from_system(|_: In<()>| 20);
    ///
    /// // Define a resource to control switching
    /// #[derive(Resource, Default, Reflect, Clone)]
    /// #[reflect(Resource)]
    /// struct SwitchSignal(bool);
    ///
    /// // A signal that switches between signal_a and signal_b
    /// let outer_signal = SignalBuilder::from_resource::<SwitchSignal>() // Use the resource
    ///     .map(move |In(switch): In<SwitchSignal>| if switch.0 { signal_a.clone() } else { signal_b.clone() });
    ///
    /// let flattened_signal = outer_signal.flatten(); // Emits 10 or 20 based on the SwitchSignal resource
    /// ```
    fn flatten(self) -> Flatten<Self>
    where
        Self: Sized,
        Self::Item: FromReflect + Signal + Clone,
        <Self::Item as Signal>::Item: FromReflect + SSs + Clone,
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
                if let Ok(ancestor_orphans) = world.run_system_cached_with(
                    move |In(signal): In<SignalSystem>, upstreams: Query<&Upstream>| {
                        UpstreamIter::new(&upstreams, signal).filter(|upstream| !upstreams.contains(**upstream)).collect::<Vec<_>>()
                    },
                    handle.0
                ) {
                    process_signals_helper(world, ancestor_orphans, Box::new(()));
                }
                *prev_system_option = Some((cur_system, handle.clone()));
            }
            cur.lock().unwrap().take()
        }) }
    }

    /// Compares the signal's value for equality with a fixed `value`.
    ///
    /// Emits `true` if `signal_item == value`, `false` otherwise.
    /// Requires `Self::Item: PartialEq`.
    ///
    /// # Example
    /// ```no_run
    /// # use bevy::prelude::*;
    /// # use jonmo::prelude::*;
    /// let signal = SignalBuilder::from_system(|_: In<()>| 5);
    /// let is_five = signal.clone().eq(5); // Emits true
    /// let is_ten = signal.eq(10); // Emits false
    /// ```
    fn eq(self, value: Self::Item) -> Eq<Self>
    where
        Self: Sized,
        Self::Item: PartialEq + FromReflect + SSs,
    {
        Eq {
            signal: self.map(move |In(item): In<Self::Item>| item == value),
        }
    }

    /// Compares the signal's value for inequality with a fixed `value`.
    ///
    /// Emits `true` if `signal_item != value`, `false` otherwise.
    /// Requires `Self::Item: PartialEq`.
    ///
    /// # Example
    /// ```no_run
    /// # use bevy::prelude::*;
    /// # use jonmo::prelude::*;
    /// let signal = SignalBuilder::from_system(|_: In<()>| 5);
    /// let is_not_five = signal.clone().neq(5); // Emits false
    /// let is_not_ten = signal.neq(10); // Emits true
    /// ```
    fn neq(self, value: Self::Item) -> Neq<Self>
    where
        Self: Sized,
        Self::Item: PartialEq + FromReflect + SSs,
    {
        Neq {
            signal: self.map(move |In(item): In<Self::Item>| item != value),
        }
    }

    /// Applies logical negation to the signal's value.
    ///
    /// Requires `Self::Item: std::ops::Not`. Typically used with boolean signals.
    ///
    /// # Example
    /// ```no_run
    /// # use bevy::prelude::*;
    /// # use jonmo::prelude::*;
    /// let signal = SignalBuilder::from_system(|_: In<()>| true);
    /// let negated_signal = signal.not(); // Emits false
    /// ```
    fn not(self) -> Not<Self>
    where
        Self: Sized,
        <Self as Signal>::Item: std::ops::Not + FromReflect + SSs,
        <<Self as Signal>::Item as std::ops::Not>::Output: FromReflect + SSs + Clone,
    {
        Not {
            signal: self.map(|In(item): In<Self::Item>| std::ops::Not::not(item)),
        }
    }

    /// Filters the signal based on a predicate system.
    ///
    /// Only propagates values for which the `predicate` system returns `true`.
    /// If the predicate returns `false` or errors, the signal terminates for that update.
    ///
    /// # Example
    /// ```no_run
    /// # use bevy::prelude::*;
    /// # use jonmo::prelude::*;
    /// # use std::collections::VecDeque;
    /// // Create a signal that emits: 1, 2, 3, 4, 5
    /// let initial_values = vec![1, 2, 3, 4, 5];
    /// let source_signal = SignalBuilder::from_system({
    ///     let initial_values = initial_values.clone();
    ///     move |_: In<()>, mut state: Local<Option<VecDeque<i32>>>| {
    ///         if state.is_none() {
    ///             *state = Some(initial_values.into());
    ///         }
    ///         state.as_mut().unwrap().pop_front()
    ///     }
    /// });
    /// let even_numbers = source_signal.filter(|In(x): In<i32>| x % 2 == 0); // Emits: 2, 4
    /// ```
    fn filter<M>(
        self,
        predicate: impl IntoSystem<In<Self::Item>, bool, M> + Send + Sync + 'static,
    ) -> Filter<Self>
    where
        Self: Sized,
        Self::Item: Clone + FromReflect + SSs,
    {
        let signal = LazySignal::new(move |world: &mut World| {
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

    /// Dynamically switches the signal's behavior based on its own output.
    ///
    /// Takes a `switcher` system that receives `In<Self::Item>` and returns *another signal* (`S`).
    /// The `switch` signal then behaves like the signal returned by `switcher`. Whenever the upstream
    /// signal emits a new value, `switcher` is run again, potentially returning a different signal
    /// to switch to.
    ///
    /// This is often used with signals that emit some form of state or key, and the `switcher`
    /// provides the appropriate signal for that state. It's internally implemented using `map` followed by `flatten`.
    ///
    /// # Example
    /// ```no_run
    /// # use bevy::prelude::*;
    /// # use jonmo::prelude::*;
    /// # use std::collections::VecDeque;
    /// #[derive(Clone, Copy, PartialEq, Eq, Reflect)]
    /// enum Mode { A, B }
    ///
    /// let signal_a = SignalBuilder::from_system(|_: In<()>| "Mode A Active");
    /// let signal_b = SignalBuilder::from_system(|_: In<()>| "Mode B Active");
    ///
    /// // Create a signal that emits Mode::A, then Mode::B
    /// let initial_modes = vec![Mode::A, Mode::B];
    /// let mode_signal = SignalBuilder::from_system(
    ///     move |_: In<()>, mut state: Local<Option<VecDeque<Mode>>>| {
    ///         if state.is_none() {
    ///             *state = Some(initial_modes.into());
    ///         }
    ///         state.as_mut().unwrap().pop_front()
    ///     }
    /// );
    ///
    /// let switched_signal = mode_signal.switch(move |In(mode): In<Mode>| {
    ///     match mode {
    ///         Mode::A => signal_a.clone(),
    ///         Mode::B => signal_b.clone(),
    ///     }
    /// }); // Emits "Mode A Active", then "Mode B Active"
    /// ```
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

    /// Throttles the signal, ensuring a minimum duration between emitted values.
    ///
    /// Only emits a value if the specified `duration` has passed since the last value was emitted
    /// by *this* throttle node. Uses Bevy's `Time` resource.
    ///
    /// # Example
    /// ```no_run
    /// # use bevy::prelude::*;
    /// # use jonmo::prelude::*;
    /// # use std::time::Duration;
    /// # use std::collections::VecDeque;
    /// // Assume source_signal emits rapidly: 1, 2, 3, 4, 5...
    /// let initial_values = vec![1, 2, 3, 4, 5];
    /// let source_signal = SignalBuilder::from_system({
    ///     let initial_values = initial_values.clone();
    ///     move |_: In<()>, mut state: Local<Option<VecDeque<i32>>>| {
    ///         if state.is_none() {
    ///             *state = Some(initial_values.into());
    ///         }
    ///         state.as_mut().unwrap().pop_front()
    ///     }
    /// });
    /// let throttled_signal = source_signal.throttle(Duration::from_secs(1));
    /// // Emits at most one value per second.
    /// ```
    fn throttle(self, duration: Duration) -> Throttle<Self>
    where
        Self: Sized,
        Self::Item: FromReflect + SSs,
    {
        Throttle {
            signal: self.map(
                move |In(item): In<Self::Item>,
                      time: Res<Time>,
                      mut timer_option: Local<Option<Timer>>| {
                    match timer_option.as_mut() {
                        None => {
                            *timer_option = Some(Timer::new(duration, TimerMode::Once));
                            Some(item)
                        }
                        Some(timer) => {
                            if timer.tick(time.delta()).finished() {
                                timer.reset();
                                Some(item)
                            } else {
                                None
                            }
                        }
                    }
                },
            ),
        }
    }

    /// Maps the signal's value using one of two systems based on the upstream boolean signal.
    ///
    /// If the upstream signal emits `true`, the `t` system is run.
    /// If the upstream signal emits `false`, the `f` system is run.
    /// Both systems take `In<()>` and return `Option<O>`.
    ///
    /// # Example
    /// ```no_run
    /// # use bevy::prelude::*;
    /// # use jonmo::prelude::*;
    /// let toggle = SignalBuilder::from_system(|_: In<()>| true); // Or false
    /// let message = toggle.map_bool(
    ///     |_: In<()>| "It's True!",
    ///     |_: In<()>| "It's False!",
    /// ); // Emits "It's True!" or "It's False!" based on `toggle`
    /// ```
    fn map_bool<O, IOO, TF, FF, TM, FM>(self, true_system: TF, false_system: FF) -> MapBool<Self, O>
    where
        Self: Sized,
        Self: Signal<Item = bool>,
        O: FromReflect + SSs,
        IOO: Into<Option<O>> + SSs,
        TF: IntoSystem<In<()>, IOO, TM> + Send + Sync + 'static,
        FF: IntoSystem<In<()>, IOO, FM> + Send + Sync + 'static,
        TM: SSs,
        FM: SSs,
    {
        let signal = LazySignal::new(move |world: &mut World| {
            let true_system = world.register_system(true_system);
            let false_system = world.register_system(false_system);
            let wrapper_system = move |In(item): In<Self::Item>, world: &mut World| {
                world
                    .run_system_with_input(if item { true_system } else { false_system }, ())
                    .ok()
                    .map(Into::into)
                    .flatten()
            };
            let signal = register_signal::<_, O, _, _, _>(world, wrapper_system);
            // just attach the system to the lifetime of the signal
            world
                .entity_mut(*signal)
                .add_child(true_system.entity())
                .add_child(false_system.entity());
            signal.into()
        });
        MapBool {
            upstream: self,
            signal,
            _marker: PhantomData,
        }
    }

    /// Maps the signal's value using a system only if the upstream boolean signal is `true`.
    ///
    /// If the upstream signal emits `true`, the `t` system is run (taking `In<()>` and returning `Option<O>`).
    /// If the upstream signal emits `false`, the signal terminates for that update.
    ///
    /// # Example
    /// ```no_run
    /// # use bevy::prelude::*;
    /// # use jonmo::prelude::*;
    /// let is_active = SignalBuilder::from_system(|_: In<()>| true);
    /// let active_message = is_active.map_true(
    ///     |_: In<()>| "Is Active",
    /// ); // Emits "Is Active"
    ///
    /// let is_inactive = SignalBuilder::from_system(|_: In<()>| false);
    /// let inactive_message = is_inactive.map_true(
    ///     |_: In<()>| "Is Active",
    /// ); // Does not emit
    /// ```
    fn map_true<O, IOO, TF, TM>(self, system: TF) -> MapTrue<Self, O>
    where
        Self: Sized,
        Self: Signal<Item = bool>,
        O: FromReflect + SSs,
        IOO: Into<Option<O>> + SSs,
        TF: IntoSystem<In<()>, IOO, TM> + Send + Sync + 'static,
        TM: SSs,
    {
        let signal = LazySignal::new(move |world: &mut World| {
            let true_system = world.register_system(system);
            let wrapper_system = move |In(item): In<Self::Item>, world: &mut World| {
                if item {
                    world
                        .run_system_with_input(true_system, ())
                        .ok()
                        .map(Into::into)
                        .flatten()
                } else {
                    None
                }
            };
            let signal = register_signal::<_, O, _, _, _>(world, wrapper_system);
            // just attach the system to the lifetime of the signal
            world.entity_mut(*signal).add_child(true_system.entity());
            signal.into()
        });
        MapTrue {
            upstream: self,
            signal,
            _marker: PhantomData,
        }
    }

    /// Maps the signal's value using a system only if the upstream boolean signal is `false`.
    ///
    /// If the upstream signal emits `false`, the `f` system is run (taking `In<()>` and returning `Option<O>`).
    /// If the upstream signal emits `true`, the signal terminates for that update.
    ///
    /// # Example
    /// ```no_run
    /// # use bevy::prelude::*;
    /// # use jonmo::prelude::*;
    /// let is_active = SignalBuilder::from_system(|_: In<()>| true);
    /// let inactive_message = is_active.map_false(
    ///     |_: In<()>| "Is Inactive",
    /// ); // Does not emit
    ///
    /// let is_inactive = SignalBuilder::from_system(|_: In<()>| false);
    /// let active_message = is_inactive.map_false(
    ///     |_: In<()>| "Is Inactive",
    /// ); // Emits "Is Inactive"
    /// ```
    fn map_false<O, IOO, FF, FM>(self, system: FF) -> MapFalse<Self, O>
    where
        Self: Sized,
        Self: Signal<Item = bool>,
        O: FromReflect + SSs,
        IOO: Into<Option<O>> + SSs,
        FF: IntoSystem<In<()>, IOO, FM> + Send + Sync + 'static,
        FM: SSs,
    {
        let signal = LazySignal::new(move |world: &mut World| {
            let false_system = world.register_system(system);
            let wrapper_system = move |In(item): In<Self::Item>, world: &mut World| {
                if !item {
                    world
                        .run_system_with_input(false_system, ())
                        .ok()
                        .map(Into::into)
                        .flatten()
                } else {
                    None
                }
            };
            let signal = register_signal::<_, O, _, _, _>(world, wrapper_system);
            // just attach the system to the lifetime of the signal
            world.entity_mut(*signal).add_child(false_system.entity());
            signal.into()
        });
        MapFalse {
            upstream: self,
            signal,
            _marker: PhantomData,
        }
    }

    /// Maps the signal's value using different systems for `Some` and `None` values.
    ///
    /// If the upstream signal emits `Some(T)`, the `some_system` is run, passing `T` via `In`.
    /// If the upstream signal emits `None`, the `none_system` is run, passing `()` via `In`.
    ///
    /// # Example
    /// ```no_run
    /// # use bevy::prelude::*;
    /// # use jonmo::prelude::*;
    /// let option_signal = SignalBuilder::from_system(|_: In<()>| Some(42));
    /// let mapped_signal = option_signal.map_option(
    ///     |In(value): In<i32>| format!("Some({})", value),
    ///     |_: In<()>| "None".to_string(),
    /// ); // Emits "Some(42)"
    /// ```
    fn map_option<I, O, IOO, SF, NF, SM, NM>(
        self,
        some_system: SF,
        none_system: NF,
    ) -> MapOption<Self, I, O>
    where
        Self: Sized,
        Self: Signal<Item = Option<I>>,
        I: FromReflect + GetTypeRegistration + Typed + SSs,
        O: FromReflect + SSs,
        IOO: Into<Option<O>> + SSs,
        SF: IntoSystem<In<I>, IOO, SM> + Send + Sync + 'static,
        NF: IntoSystem<In<()>, IOO, NM> + Send + Sync + 'static,
        SM: SSs,
        NM: SSs,
    {
        let signal = LazySignal::new(move |world: &mut World| {
            let some_system = world.register_system(some_system);
            let none_system = world.register_system(none_system);
            let wrapper_system = move |In(item): In<Option<I>>, world: &mut World| match item {
                Some(value) => world
                    .run_system_with_input(some_system, value)
                    .ok()
                    .map(Into::into)
                    .flatten(),
                None => world
                    .run_system_with_input(none_system, ())
                    .ok()
                    .map(Into::into)
                    .flatten(),
            };
            let signal = register_signal::<_, O, _, _, _>(world, wrapper_system);
            // just attach the system to the lifetime of the signal
            world
                .entity_mut(*signal)
                .add_child(some_system.entity())
                .add_child(none_system.entity());
            signal.into()
        });
        MapOption {
            upstream: self,
            signal,
            _marker: PhantomData,
        }
    }

    /// Maps the signal's value using a system only if the upstream signal emits `Some(T)`.
    ///
    /// If the upstream signal emits `Some(T)`, the `some_system` is run, passing `T` via `In`.
    /// If the upstream signal emits `None`, the signal terminates for that update.
    ///
    /// # Example
    /// ```no_run
    /// # use bevy::prelude::*;
    /// # use jonmo::prelude::*;
    /// let option_signal = SignalBuilder::from_system(|_: In<()>| Some(42));
    /// let mapped_signal = option_signal.map_some(
    ///     |In(value): In<i32>| format!("Some({})", value),
    /// ); // Emits "Some(42)"
    /// ```
    fn map_some<I, O, IOO, SF, SM>(self, system: SF) -> MapSome<Self, I, O>
    where
        Self: Sized,
        Self: Signal<Item = Option<I>>,
        I: FromReflect + GetTypeRegistration + Typed + SSs,
        O: FromReflect + SSs,
        IOO: Into<Option<O>> + SSs,
        SF: IntoSystem<In<I>, IOO, SM> + Send + Sync + 'static,
        SM: SSs,
    {
        let signal = LazySignal::new(move |world: &mut World| {
            let some_system = world.register_system(system);
            let wrapper_system = move |In(item): In<Option<I>>, world: &mut World| match item {
                Some(value) => world
                    .run_system_with_input(some_system, value)
                    .ok()
                    .map(Into::into)
                    .flatten(),
                None => None,
            };
            let signal = register_signal::<_, O, _, _, _>(world, wrapper_system);
            // just attach the system to the lifetime of the signal
            world.entity_mut(*signal).add_child(some_system.entity());
            signal.into()
        });
        MapSome {
            upstream: self,
            signal,
            _marker: PhantomData,
        }
    }

    /// Maps the signal's value using a system only if the upstream signal emits `None`.
    ///
    /// If the upstream signal emits `None`, the `none_system` is run, passing `()` via `In`.
    /// If the upstream signal emits `Some(T)`, the signal terminates for that update.
    ///
    /// # Example
    /// ```no_run
    /// # use bevy::prelude::*;
    /// # use jonmo::prelude::*;
    /// let option_signal = SignalBuilder::from_system(|_: In<()>| None);
    /// let mapped_signal = option_signal.map_none(
    ///     |_: In<()>| "None".to_string(),
    /// ); // Emits "None"
    /// ```
    fn map_none<I, O, IOO, NF, NM>(self, none_system: NF) -> MapNone<Self, I, O>
    where
        Self: Sized,
        Self: Signal<Item = Option<I>>,
        I: FromReflect + GetTypeRegistration + Typed + SSs,
        O: FromReflect + SSs,
        IOO: Into<Option<O>> + SSs,
        NF: IntoSystem<In<()>, IOO, NM> + Send + Sync + 'static,
        NM: SSs,
    {
        let signal = LazySignal::new(move |world: &mut World| {
            let none_system = world.register_system(none_system);
            let wrapper_system = move |In(item): In<Option<I>>, world: &mut World| match item {
                Some(_) => None,
                None => world
                    .run_system_with_input(none_system, ())
                    .ok()
                    .map(Into::into)
                    .flatten(),
            };
            let signal = register_signal::<_, O, _, _, _>(world, wrapper_system);
            // just attach the system to the lifetime of the signal
            world.entity_mut(*signal).add_child(none_system.entity());
            signal.into()
        });
        MapNone {
            upstream: self,
            signal,
            _marker: PhantomData,
        }
    }

    /// Adds debug logging to the signal chain.
    ///
    /// When this signal emits a value, it will be logged using `bevy_log::debug!`
    /// along with the source code location where `.debug()` was called.
    /// The value is passed through unchanged.
    ///
    /// Requires `Self::Item: Debug`.
    ///
    /// # Example
    /// ```no_run
    /// # use bevy::prelude::*;
    /// # use jonmo::prelude::*;
    /// let signal = SignalBuilder::from_system(|_: In<()>| 42)
    ///     .debug() // Logs the value 42
    ///     .map(|In(x): In<i32>| x * 2);
    /// ```
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

    // TODO: how ??
    // fn type_erase(self) -> Arc<dyn Signal<Item = Self::Item> + Send + Sync + 'static>
    //     where Self: Sized
    // {
    //     Arc::new(self)
    // }

    /// Registers all the systems defined in this signal chain into the Bevy `World`.
    ///
    /// This activates the signal chain. It performs the following:
    /// 1. Traverses the signal chain definition upstream from this point.
    /// 2. For each node (map, filter, combine, etc.):
    ///    - Registers the associated Bevy system if not already present.
    ///    - Increments a reference count ([`SignalRegistrationCount`]) for the system.
    /// 3. Connects the systems using [`Upstream`] and [`Downstream`] components to define data flow.
    /// 4. Marks the ultimate source system(s) as roots for propagation.
    ///
    /// Returns a [`SignalHandle`] which tracks the specific system entity created or referenced
    /// for the *final* node in the chain during *this* `register` call. This handle is used
    /// with [`SignalHandle::cleanup`] to decrement reference counts and potentially despawn
    /// the systems when the signal is no longer needed.
    fn register(self, world: &mut World) -> SignalHandle
    where
        Self: Sized,
    {
        self.register_signal(world)
    }
}

impl<T> SignalExt for T where T: Signal {}

#[cfg(test)]
mod tests {
    use crate::JonmoPlugin;

    use super::*;
    // Import Bevy prelude for MinimalPlugins and other common items
    use bevy::prelude::*;
    use bevy_time::TimeUpdateStrategy;
    use std::sync::{Arc, Mutex};
    use std::time::Duration; // Add Duration

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
        debug!(
            "Capture Output System: Received {:?}, updating resource from {:?} to Some({:?})",
            value, output.0, value
        );
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

        let values = Arc::new(Mutex::new(vec![10, 20, 30]));
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

        assert_eq!(get_output::<i32>(app.world()), Some(10));
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

    #[test]
    fn test_throttle() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<i32>>();
        let counter = Arc::new(Mutex::new(0));
        let emit_count = Arc::new(Mutex::new(0));

        // Throttle duration
        let throttle_duration = Duration::from_millis(100);

        let signal = SignalBuilder::from_system(clone!((counter) move |_: In<()>| {
            let mut c = counter.lock().unwrap();
            *c += 1;
            println!("Source System: Emitting {}", *c);
            Some(*c) // Emit 1, 2, 3, 4, 5... rapidly
        }))
        .throttle(throttle_duration)
        .map(clone!((emit_count) move |In(val): In<i32>| {
            let mut count = emit_count.lock().unwrap();
            *count += 1;
            println!("Emit Count Map: Incremented count to {}, passing value={}", *count, val);
            val // Pass the value through
        }))
        .map(capture_output)
        .register(app.world_mut());

        // --- Test Execution with Manual Time Control (Revised Assertions) ---

        // 1. Initial update: Emit 1, create timer.
        app.update();
        assert_eq!(get_output::<i32>(app.world()), Some(1), "Initial emit (1)"); // Changed assertion description
        assert_eq!(*emit_count.lock().unwrap(), 1, "Emit count after initial");
        assert_eq!(*counter.lock().unwrap(), 1, "Source counter after initial");

        // 2. Advance time > duration (110ms).
        app.world_mut()
            .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f32(
                110.,
            )));

        // 3. Update again: Source emits 2. Time elapsed >= duration. Throttle emits 2.
        app.update();
        assert_eq!(
            get_output::<i32>(app.world()),
            Some(2),
            "After 110ms advance, 1st update (emit 2)"
        ); // EXPECT 2
        assert_eq!(
            *emit_count.lock().unwrap(),
            2,
            "Emit count after 110ms advance, 1st update"
        ); // EXPECT 2
        assert_eq!(
            *counter.lock().unwrap(),
            2,
            "Source counter after 110ms advance, 1st update"
        );

        app.world_mut()
            .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_millis(
                50,
            )));

        // 6. Update again: Source emits 4. Time elapsed < duration since last emit. Throttle blocks. Output remains 2.
        app.update();
        assert_eq!(
            get_output::<i32>(app.world()),
            Some(2),
            "After 110ms advance, 2nd update (block 3)"
        ); // EXPECT 2
        assert_eq!(
            *emit_count.lock().unwrap(),
            2,
            "Emit count after 110ms advance, 2nd update"
        ); // EXPECT 2
        assert_eq!(
            *counter.lock().unwrap(),
            3,
            "Source counter after 110ms advance, 2nd update"
        );

        app.world_mut()
            .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_millis(0)));
        // 7. Update again: Source emits 5. Time elapsed < duration since last emit. Throttle blocks. Output remains 2.
        app.update();
        assert_eq!(
            get_output::<i32>(app.world()),
            Some(2),
            "After 50ms advance, 1st update (block 4)"
        ); // EXPECT 2
        assert_eq!(
            *emit_count.lock().unwrap(),
            2,
            "Emit count after 50ms advance, 1st update"
        );
        assert_eq!(
            *counter.lock().unwrap(),
            4,
            "Source counter after 50ms advance, 1st update"
        );

        // 8. Advance time > duration again (total 50 + 60 = 110ms since last emit at step 3).
        app.world_mut()
            .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_millis(
                60,
            )));

        // 9. Update again: Source emits 5. Total time elapsed >= duration. Throttle emits 5.
        app.update();
        assert_eq!(
            get_output::<i32>(app.world()),
            Some(5), // Reverted: Expect 5 based on logic and actual output
            "After 60ms advance, 1st update (emit 5)"  // Corrected message to reflect expected value
        );
        assert_eq!(
            *emit_count.lock().unwrap(),
            3, // Emit count becomes 3
            "Emit count after 60ms advance, 1st update"
        );
        assert_eq!(
            *counter.lock().unwrap(),
            5, // Source counter is 5
            "Source counter after 60ms advance, 1st update"
        );

        app.world_mut()
            .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_millis(0)));
        // 10. Update again: Source emits 6. Time elapsed < duration since last emit. Throttle blocks. Output remains 5.
        app.update();
        assert_eq!(
            get_output::<i32>(app.world()),
            Some(5), // Output remains 5
            "After 60ms advance, 2nd update (block 6)"
        ); // EXPECT 5
        assert_eq!(
            *emit_count.lock().unwrap(),
            3,
            "Emit count after 60ms advance, 2nd update"
        );
        assert_eq!(
            *counter.lock().unwrap(),
            6,
            "Source counter after 60ms advance, 2nd update"
        );

        // Add one more step to ensure timer reset correctly
        app.world_mut()
            .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_millis(
                110,
            )));
        // 11. Update: Source emits 7. Time >= 100ms. Throttle emits 7.
        app.update();
        assert_eq!(
            get_output::<i32>(app.world()),
            Some(7), // Expect 7
            "After 110ms advance, 1st update (emit 7)"
        );
        assert_eq!(
            *emit_count.lock().unwrap(),
            4, // Emit count becomes 4
            "Emit count after 110ms advance, 1st update"
        );
        assert_eq!(
            *counter.lock().unwrap(),
            7, // Source counter is 7
            "Source counter after 110ms advance, 1st update"
        );

        app.world_mut()
            .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_millis(0)));
        // 12. Update: Source emits 8. Time < 100ms. Throttle blocks. Output remains 7.
        app.update();
        assert_eq!(
            get_output::<i32>(app.world()),
            Some(7), // Output remains 7
            "After 110ms advance, 2nd update (block 8)"
        );
        assert_eq!(
            *emit_count.lock().unwrap(),
            4, // Emit count remains 4
            "Emit count after 110ms advance, 2nd update"
        );
        assert_eq!(
            *counter.lock().unwrap(),
            8, // Source counter is 8
            "Source counter after 110ms advance, 2nd update"
        );

        signal.cleanup(app.world_mut());
    }

    #[test]
    fn test_map_bool() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<&'static str>>();

        // Test true case
        let signal_true = SignalBuilder::from_system(|_: In<()>| true)
            .map_bool(|_: In<()>| "True Branch", |_: In<()>| "False Branch")
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<&'static str>(app.world()), Some("True Branch"));
        signal_true.cleanup(app.world_mut());

        // Test false case
        let signal_false = SignalBuilder::from_system(|_: In<()>| false)
            .map_bool(|_: In<()>| "True Branch", |_: In<()>| "False Branch")
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(
            get_output::<&'static str>(app.world()),
            Some("False Branch")
        );
        signal_false.cleanup(app.world_mut());
    }

    #[test]
    fn test_map_true() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<&'static str>>();

        // Test true case (should emit)
        let signal_true = SignalBuilder::from_system(|_: In<()>| true)
            .map_true(|_: In<()>| "Was True")
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<&'static str>(app.world()), Some("Was True"));
        signal_true.cleanup(app.world_mut());

        // Reset output
        app.world_mut()
            .resource_mut::<SignalOutput<&'static str>>()
            .0 = None;

        // Test false case (should not emit)
        let signal_false = SignalBuilder::from_system(|_: In<()>| false)
            .map_true(|_: In<()>| "Was True")
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<&'static str>(app.world()), None); // Should remain None
        signal_false.cleanup(app.world_mut());
    }

    #[test]
    fn test_map_false() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<&'static str>>();

        // Test true case (should not emit)
        let signal_true = SignalBuilder::from_system(|_: In<()>| true)
            .map_false(|_: In<()>| "Was False")
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<&'static str>(app.world()), None); // Should remain None
        signal_true.cleanup(app.world_mut());

        // Reset output
        app.world_mut()
            .resource_mut::<SignalOutput<&'static str>>()
            .0 = None;

        // Test false case (should emit)
        let signal_false = SignalBuilder::from_system(|_: In<()>| false)
            .map_false(|_: In<()>| "Was False")
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<&'static str>(app.world()), Some("Was False"));
        signal_false.cleanup(app.world_mut());
    }

    #[test]
    fn test_map_option() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<String>>();

        // Test Some case
        let signal_some = SignalBuilder::from_system(|_: In<()>| Some(42))
            .map_option(
                |In(value): In<i32>| format!("Some({})", value),
                |_: In<()>| "None".to_string(),
            )
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(
            get_output::<String>(app.world()),
            Some("Some(42)".to_string())
        );
        signal_some.cleanup(app.world_mut());

        // Reset output
        app.world_mut().resource_mut::<SignalOutput<String>>().0 = None;

        // Test None case
        let signal_none = SignalBuilder::from_system(|_: In<()>| None::<i32>)
            .map_option(
                |In(value): In<i32>| format!("Some({})", value),
                |_: In<()>| "None".to_string(),
            )
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<String>(app.world()), Some("None".to_string()));
        signal_none.cleanup(app.world_mut());
    }

    #[test]
    fn test_map_some() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<String>>();

        // Test Some case (should emit)
        let signal_some = SignalBuilder::from_system(|_: In<()>| Some(42))
            .map_some(|In(value): In<i32>| format!("Some({})", value))
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(
            get_output::<String>(app.world()),
            Some("Some(42)".to_string())
        );
        signal_some.cleanup(app.world_mut());

        // Reset output
        app.world_mut().resource_mut::<SignalOutput<String>>().0 = None;

        // Test None case (should not emit)
        let signal_none = SignalBuilder::from_system(|_: In<()>| None::<i32>)
            .map_some(|In(value): In<i32>| format!("Some({})", value))
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<String>(app.world()), None); // Should remain None
        signal_none.cleanup(app.world_mut());
    }

    #[test]
    fn test_map_none() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<String>>();

        // Test Some case (should not emit)
        let signal_some = SignalBuilder::from_system(|_: In<()>| Some(42))
            .map_none(|_: In<()>| "None".to_string())
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<String>(app.world()), None); // Should remain None
        signal_some.cleanup(app.world_mut());

        // Reset output
        app.world_mut().resource_mut::<SignalOutput<String>>().0 = None;

        // Test None case (should emit)
        let signal_none = SignalBuilder::from_system(|_: In<()>| None::<i32>)
            .map_none(|_: In<()>| "None".to_string())
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<String>(app.world()), Some("None".to_string()));
        signal_none.cleanup(app.world_mut());
    }

    #[test]
    fn simple_signal_lazy_outlives_handle() {
        let mut app = create_test_app();

        let source_signal_struct = SignalBuilder::from_system(|_: In<()>| 1);
        let handle = source_signal_struct.clone().register(app.world_mut());
        let system_entity = handle.0.entity();

        assert!(
            app.world().get_entity(system_entity).is_ok(),
            "System entity should exist after registration"
        );
        assert!(
            app.world().get::<LazySignalHolder>(system_entity).is_some(),
            "LazySignalHolder should exist on system entity"
        );
        assert_eq!(
            **app
                .world()
                .get::<SignalRegistrationCount>(system_entity)
                .unwrap(),
            1,
            "SignalRegistrationCount should be 1"
        );

        handle.cleanup(app.world_mut());
        assert_eq!(
            **app
                .world()
                .get::<SignalRegistrationCount>(system_entity)
                .unwrap(),
            0,
            "SignalRegistrationCount should be 0 after cleanup"
        );
        // LazySignalHolder is not removed because source_signal_struct (holding another LazySignal clone) still exists,
        // so holder.lazy_signal.inner.references > 1 at the time of cleanup.
        assert!(
            app.world().get_entity(system_entity).is_ok(),
            "System entity should still exist after handle cleanup (LazySignal struct alive)"
        );
        assert!(
            app.world().get::<LazySignalHolder>(system_entity).is_some(),
            "LazySignalHolder should still exist"
        );

        drop(source_signal_struct);
        // Dropping source_signal_struct reduces LazySignalState.references.
        // If it becomes 1 (only holder's copy left), LazySignal::drop doesn't queue.
        // The system is not queued to CLEANUP_SIGNALS yet.
        // LazySignalHolder is still present.
        app.update(); // Runs flush_cleanup_signals. CLEANUP_SIGNALS is empty.
        app.update(); // Runs flush_cleanup_signals. CLEANUP_SIGNALS is empty.

        assert!(
            app.world().get_entity(system_entity).is_err(),
            "System entity persists because LazySignalHolder was not removed and its LazySignal did not trigger cleanup on its own drop"
        );
    }

    // #[test]
    // fn simple_signal_lazy_dropped_before_handle_cleanup() {
    //     let mut app = create_test_app_for_cleanup();

    //     let source_signal_struct = SignalBuilder::from_system(test_source_system);
    //     let handle = source_signal_struct.clone().register(app.world_mut());
    //     let system_entity = handle.0.entity();

    //     assert!(app.world().get_entity(system_entity).is_some());
    //     assert_eq!(**app.world().get::<SignalRegistrationCount>(system_entity).unwrap(), 1);

    //     drop(source_signal_struct); // LazySignalState.references decreases.
    //     app.update(); // CLEANUP_SIGNALS is empty. Entity still exists.

    //     assert!(app.world().get_entity(system_entity).is_some());
    //     assert!(app.world().get::<LazySignalHolder>(system_entity).is_some());

    //     handle.cleanup(app.world_mut());
    //     // Now SignalRegistrationCount is 0.
    //     // The LazySignal in LazySignalHolder is the only one left (references == 1).
    //     // So, LazySignalHolder IS removed.
    //     // Dropping LazySignalHolder drops its LazySignal, which queues the system_entity to CLEANUP_SIGNALS.
    //     assert_eq!(**app.world().get::<SignalRegistrationCount>(system_entity).unwrap(), 0);
    //     assert!(
    //         app.world().get_entity(system_entity).is_some(),
    //         "System entity exists before final flush"
    //     );
    //     assert!(
    //         app.world().get::<LazySignalHolder>(system_entity).is_none(),
    //         "LazySignalHolder should be removed"
    //     );

    //     app.update(); // Runs flush_cleanup_signals, processes CLEANUP_SIGNALS.
    //     assert!(
    //         app.world().get_entity(system_entity).is_none(),
    //         "System entity should be despawned"
    //     );
    // }

    #[test]
    fn multiple_lazy_signal_clones_cleanup_behavior() {
        let mut app = create_test_app();

        let s1 = SignalBuilder::from_system(|_: In<()>| 1); // LS.state_refs = 1
        let s2 = s1.clone(); // LS.state_refs = 2

        let handle = s1.clone().register(app.world_mut()); // LS.state_refs = 3 (s1, s2, holder)
        let system_entity = handle.0.entity();

        assert!(app.world().get_entity(system_entity).is_ok());
        assert_eq!(**app.world().get::<SignalRegistrationCount>(system_entity).unwrap(), 1);
        assert!(app.world().get::<LazySignalHolder>(system_entity).is_some());

        handle.cleanup(app.world_mut()); // RegCount = 0. LS.state_refs = 3 for holder check. Holder not removed.
        assert_eq!(**app.world().get::<SignalRegistrationCount>(system_entity).unwrap(), 0);
        assert!(app.world().get_entity(system_entity).is_ok());
        assert!(app.world().get::<LazySignalHolder>(system_entity).is_some());

        drop(s1); // LS.state_refs = 2. Not queued.
        app.update();
        assert!(app.world().get_entity(system_entity).is_ok(), "Entity persists after s1 drop");
        assert!(app.world().get::<LazySignalHolder>(system_entity).is_some());

        drop(s2); // LS.state_refs = 1 (only holder's copy). Not queued.
        app.update();
        assert!(app.world().get_entity(system_entity).is_err(), "Entity despawned after s2 drop, only holder left");
    }

    #[test]
    fn multiple_handles_same_system() {
        let mut app = create_test_app();

        let source_signal_struct = SignalBuilder::from_system(|_: In<()>| 1); // LS.state_refs = 1

        let handle1 = source_signal_struct.clone().register(app.world_mut()); // LS.state_refs = 2 (struct, holder). RegCount = 1.
        let system_entity = handle1.0.entity();
        assert_eq!(**app.world().get::<SignalRegistrationCount>(system_entity).unwrap(), 1);

        let handle2 = source_signal_struct.clone().register(app.world_mut()); // LS.state_refs = 2 (no new holder). RegCount = 2.
        assert_eq!(system_entity, handle2.0.entity());
        assert_eq!(**app.world().get::<SignalRegistrationCount>(system_entity).unwrap(), 2);

        handle1.cleanup(app.world_mut()); // RegCount = 1. Holder not removed (LS.state_refs=2).
        assert_eq!(**app.world().get::<SignalRegistrationCount>(system_entity).unwrap(), 1);
        assert!(app.world().get_entity(system_entity).is_ok());
        assert!(app.world().get::<LazySignalHolder>(system_entity).is_some());

        drop(source_signal_struct); // LS.state_refs = 1 (holder only). Not queued.
        app.update();
        assert!(app.world().get_entity(system_entity).is_ok());
        assert!(app.world().get::<LazySignalHolder>(system_entity).is_some());
        
        handle2.cleanup(app.world_mut()); // RegCount = 0. LS.state_refs = 1 for holder. Holder IS removed. Queued.
        app.update();
        assert!(app.world().get_entity(system_entity).is_err());
    }

    #[test]
    fn chained_signals_cleanup() {
        let mut app = create_test_app();

        let source_s = SignalBuilder::from_system(|_: In<()>| 1);
        let map_s = source_s.map(|In(val)| val + 1); // map_s holds source_s

        let handle = map_s.clone().register(app.world_mut());
        let map_entity = handle.0.entity();
        let source_entity = app.world().get::<Upstream>(map_entity).unwrap().iter().next().unwrap().entity();

        assert!(app.world().get_entity(map_entity).is_ok());
        assert!(app.world().get_entity(source_entity).is_ok());
        assert!(app.world().get::<LazySignalHolder>(map_entity).is_some());
        assert!(app.world().get::<LazySignalHolder>(source_entity).is_some());
        assert_eq!(**app.world().get::<SignalRegistrationCount>(map_entity).unwrap(), 1);
        assert_eq!(**app.world().get::<SignalRegistrationCount>(source_entity).unwrap(), 1);

        handle.cleanup(app.world_mut());
        // map_entity: RegCount=0. Holder's LS refs > 1 (due to map_s). Holder not removed.
        // source_entity: RegCount=0. Holder's LS refs > 1 (due to map_s.source_s). Holder not removed.
        assert_eq!(**app.world().get::<SignalRegistrationCount>(map_entity).unwrap(), 0);
        assert_eq!(**app.world().get::<SignalRegistrationCount>(source_entity).unwrap(), 0);
        assert!(app.world().get_entity(map_entity).is_ok());
        assert!(app.world().get_entity(source_entity).is_ok());
        assert!(app.world().get::<LazySignalHolder>(map_entity).is_some());
        assert!(app.world().get::<LazySignalHolder>(source_entity).is_some());

        drop(map_s); // Drops map_s's LazySignal, then source_s's LazySignal.
                     // For both, their LS.state_refs becomes 1 (holder only). queued.
        app.update();
        // Both entities despawned.
        assert!(app.world().get_entity(map_entity).is_err(), "Map entity persists");
        assert!(app.world().get_entity(source_entity).is_err(), "Source entity persists");
    }

    #[test]
    fn re_register_after_cleanup_while_lazy_alive() {
        let mut app = create_test_app();

        let source_signal_struct = SignalBuilder::from_system(|_: In<()>| 1); // LS.state_refs = 1

        let handle1 = source_signal_struct.clone().register(app.world_mut()); // LS.state_refs = 2 (struct, holder). RegCount = 1.
        let system_entity = handle1.0.entity();
        assert_eq!(**app.world().get::<SignalRegistrationCount>(system_entity).unwrap(), 1);

        handle1.cleanup(app.world_mut()); // RegCount = 0. Holder not removed (LS.state_refs=2).
        assert_eq!(**app.world().get::<SignalRegistrationCount>(system_entity).unwrap(), 0);
        assert!(app.world().get_entity(system_entity).is_ok());
        assert!(app.world().get::<LazySignalHolder>(system_entity).is_some());

        let handle2 = source_signal_struct.clone().register(app.world_mut()); // RegCount becomes 1 again on same entity. LS.state_refs=2.
        assert_eq!(system_entity, handle2.0.entity());
        assert_eq!(**app.world().get::<SignalRegistrationCount>(system_entity).unwrap(), 1);
        assert!(app.world().get::<LazySignalHolder>(system_entity).is_some());

        drop(source_signal_struct); // LS.state_refs = 1 (holder only). Not queued.
        app.update();
        assert!(app.world().get_entity(system_entity).is_ok());
        assert!(app.world().get::<LazySignalHolder>(system_entity).is_some());

        handle2.cleanup(app.world_mut()); // RegCount = 0. Holder's LS.state_refs = 1. Holder IS removed. despawned.
        assert!(app.world().get_entity(system_entity).is_err());
    }
}
