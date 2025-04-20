use bevy_ecs::prelude::*; // Removed unused system::SystemId
use bevy_reflect::{FromReflect, GetTypeRegistration, Typed, prelude::*}; // Add reflection traits
use std::{
    fmt,
    marker::PhantomData,
    ops::Deref,
    sync::{Arc, OnceLock, RwLock, RwLockReadGuard, RwLockWriteGuard}, // Added RwLock and guards
};

use crate::{
    signal::SignalHandle, // Use SignalHandle::new constructor
    tree::{UntypedSystemId, pipe_signal, register_signal}, // Removed mark_signal_root // Removed unused SignalNodeMetadata
};
// Removed unused RunSystemOnce import
// Removed unused crate::identity import

//-------------------------------------------------------------------------------------------------
// VecDiff - Based on futures-signals VecDiff
//-------------------------------------------------------------------------------------------------

/// Describes the changes to a `Vec`.
///
/// This is used by [`SignalVec`] to efficiently represent changes.
#[derive(Reflect)] // Removed PartialEq, removed #[reflect(PartialEq)]
pub enum VecDiff<T>
where
    T: Reflect + FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static, // Removed PartialEq
{
    // Add PartialEq bound
    /// Replaces the entire contents of the `Vec`.
    Replace {
        /// The new values for the vector.
        values: Vec<T>,
    },
    /// Inserts a new item at the `index`.
    InsertAt {
        /// The index where the value should be inserted.
        index: usize,
        /// The value to insert.
        value: T,
    },
    /// Updates the item at the `index`.
    UpdateAt {
        /// The index of the value to update.
        index: usize,
        /// The new value.
        value: T,
    },
    /// Removes the item at the `index`.
    RemoveAt {
        /// The index of the value to remove.
        index: usize,
    },
    /// Moves the item at `old_index` to `new_index`.
    Move {
        /// The original index of the item.
        old_index: usize,
        /// The new index for the item.
        new_index: usize,
    },
    /// Appends a new item to the end of the `Vec`.
    Push {
        /// The value to append.
        value: T,
    },
    /// Removes the last item from the `Vec`.
    Pop,
    /// Removes all items from the `Vec`.
    Clear,
    // NOTE: futures-signals has Truncate, but it's less common and can be represented by multiple RemoveAt/Pop.
}

// Manual Debug implementation as derive(Debug) requires T: Debug
impl<T> fmt::Debug for VecDiff<T>
where
    T: fmt::Debug + Reflect + FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static, // Removed PartialEq
{
    // Add PartialEq bound
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Replace { values } => f.debug_struct("Replace").field("values", values).finish(),
            Self::InsertAt { index, value } => f
                .debug_struct("InsertAt")
                .field("index", index)
                .field("value", value)
                .finish(),
            Self::UpdateAt { index, value } => f
                .debug_struct("UpdateAt")
                .field("index", index)
                .field("value", value)
                .finish(),
            Self::RemoveAt { index } => f.debug_struct("RemoveAt").field("index", index).finish(),
            Self::Move {
                old_index,
                new_index,
            } => f
                .debug_struct("Move")
                .field("old_index", old_index)
                .field("new_index", new_index)
                .finish(),
            Self::Push { value } => f.debug_struct("Push").field("value", value).finish(),
            Self::Pop => f.write_str("Pop"),
            Self::Clear => f.write_str("Clear"),
        }
    }
}

// Manual Clone implementation as derive(Clone) requires T: Clone
impl<T> Clone for VecDiff<T>
where
    T: Clone + Reflect + FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static, // Removed PartialEq
{
    // Add PartialEq bound
    #[inline]
    fn clone(&self) -> Self {
        match self {
            Self::Replace { values } => Self::Replace {
                values: values.clone(),
            },
            Self::InsertAt { index, value } => Self::InsertAt {
                index: *index,
                value: value.clone(),
            },
            Self::UpdateAt { index, value } => Self::UpdateAt {
                index: *index,
                value: value.clone(),
            },
            Self::RemoveAt { index } => Self::RemoveAt { index: *index },
            Self::Move {
                old_index,
                new_index,
            } => Self::Move {
                old_index: *old_index,
                new_index: *new_index,
            },
            Self::Push { value } => Self::Push {
                value: value.clone(),
            },
            Self::Pop => Self::Pop,
            Self::Clear => Self::Clear,
        }
    }
}

//-------------------------------------------------------------------------------------------------
// Source System Logic (Reverted)
//-------------------------------------------------------------------------------------------------

// Removed SignalVecSourceState component
// Removed SignalVecDirtyFlag component
// Removed state_emitter_system function

// Removed unused identity_batch_system function

// --- Consolidated SignalVec Trait ---

/// Represents a `Vec` that changes over time, yielding [`VecDiff<T>`] and handling registration.
///
/// Instead of yielding the entire `Vec` with each change, it yields [`VecDiff<T>`]
/// describing the change. This trait combines the public concept with internal registration.
pub trait SignalVec: Clone + Send + Sync + 'static {
    /// The type of items in the vector.
    type Item: Reflect + FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static;

    /// Registers the systems associated with this node and its predecessors in the `World`.
    /// Returns a [`SignalHandle`] containing the entities of *all* systems
    /// registered or reference-counted during this specific registration call instance.
    /// **Note:** This method is intended for internal use by the signal combinators and registration process.
    fn register(&self, world: &mut World) -> SignalHandle;
}

/// A source node for a `SignalVec` chain. Holds the entity ID of the registered source system.
#[derive(Clone)] // Clone is fine, just copies the Entity ID
pub struct SourceVec<T>
where
    T: Reflect + FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static,
{
    /// The entity ID of the Bevy system that acts as the source for this signal.
    pub(crate) entity: Entity,
    _marker: PhantomData<T>,
}

// Implement SignalVec for SourceVec<T>
impl<T> SignalVec for SourceVec<T>
where
    T: Clone + Reflect + FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static,
{
    type Item = T;

    /// Registers the systems associated with this node. For a SourceVec, it's already registered.
    fn register(&self, _world: &mut World) -> SignalHandle {
        // The system is already registered (e.g., by MutableVec::signal_vec or SignalVecBuilder::from_system)
        // We just need to return its entity ID wrapped in a handle.
        SignalHandle::new(vec![self.entity]) // Return SignalHandle
    }
}

type MapVecRegisterFn = dyn Fn(
        &mut World,
        UntypedSystemId, // Previous system's entity ID
    ) -> UntypedSystemId // Registered map system's entity ID
    + Send
    + Sync
    + 'static;

/// A map node in a `SignalVec` chain.
pub struct MapVec<Prev, U>
where
    Prev: SignalVec, // Use consolidated SignalVec trait
    U: Reflect + FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static,
    <Prev as SignalVec>::Item:
        Reflect + FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static,
{
    pub(crate) prev_signal: Prev,
    pub(crate) register_fn: Arc<MapVecRegisterFn>,
    _marker: PhantomData<U>,
}

/// A terminal node in a `SignalVec` chain that executes a system for each batch.
pub struct ForEachVec<Prev>
where
    Prev: SignalVec, // Use consolidated SignalVec trait
    <Prev as SignalVec>::Item:
        Reflect + FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static,
{
    pub(crate) prev_signal: Prev,
    pub(crate) register_fn: Arc<MapVecRegisterFn>, // Re-use MapVecRegisterFn type alias
}

impl<Prev> Clone for ForEachVec<Prev>
where
    Prev: SignalVec + Clone, // Use consolidated SignalVec trait
    <Prev as SignalVec>::Item:
        Reflect + FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        Self {
            prev_signal: self.prev_signal.clone(),
            register_fn: self.register_fn.clone(),
        }
    }
}

// Implement SignalVec for ForEachVec<Prev>
// Note: Item is (), as this is a terminal node.
impl<Prev> SignalVec for ForEachVec<Prev>
where
    Prev: SignalVec, // Use consolidated SignalVec trait
    <Prev as SignalVec>::Item:
        Reflect + FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static,
{
    type Item = (); // Terminal node, effectively produces no further signal items

    fn register(&self, world: &mut World) -> SignalHandle {
        let prev_handle = self.prev_signal.register(world);
        let mut all_ids = prev_handle.system_ids().to_vec();

        if let Some(&prev_last_id_entity) = all_ids.last() {
            let new_system_entity = (self.register_fn)(world, prev_last_id_entity);
            all_ids.push(new_system_entity);
        } else {
            bevy_log::error!("ForEachVec signal parent registration returned empty ID list.");
        }
        SignalHandle::new(all_ids)
    }
}

impl<Prev, U> Clone for MapVec<Prev, U>
where
    Prev: SignalVec + Clone, // Use consolidated SignalVec trait
    U: Reflect + FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static,
    <Prev as SignalVec>::Item:
        Reflect + FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        Self {
            prev_signal: self.prev_signal.clone(),
            register_fn: self.register_fn.clone(),
            _marker: PhantomData,
        }
    }
}

// Implement SignalVec for MapVec<Prev, U>
impl<Prev, U> SignalVec for MapVec<Prev, U>
where
    Prev: SignalVec, // Use consolidated SignalVec trait
    U: Reflect + FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static,
    <Prev as SignalVec>::Item:
        Reflect + FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static,
{
    type Item = U;

    fn register(&self, world: &mut World) -> SignalHandle {
        let prev_handle = self.prev_signal.register(world); // Returns SignalHandle
        let mut all_ids = prev_handle.system_ids().to_vec(); // Get Vec<UntypedSystemId>

        if let Some(&prev_last_id_entity) = all_ids.last() {
            let new_system_entity = (self.register_fn)(world, prev_last_id_entity);
            all_ids.push(new_system_entity);
        } else {
            bevy_log::error!("MapVec signal parent registration returned empty ID list.");
        }
        SignalHandle::new(all_ids) // Return new SignalHandle with all IDs
    }
}

//-------------------------------------------------------------------------------------------------
// SignalVecBuilder and SignalVecExt
//-------------------------------------------------------------------------------------------------

/// Provides static methods for creating new `SignalVec` chains.
pub struct SignalVecBuilder;

impl SignalVecBuilder {
    // *** Removing from_system for now ***
    // pub fn from_system<T, M, F>(system: F) -> SourceVec<T> ...
    // todo!("SignalVecBuilder::from_system needs API revision to handle World access for registration");
}

/// A generic system that acts as a termination point in the signal graph
/// for the map cleanup strategy. It takes a Vec<VecDiff<U>> and returns Option<T>.
fn terminator_system<T, U>(
    In(_diff_batch): In<Vec<VecDiff<U>>>, // Input is Vec<VecDiff<U>>
) -> Option<T>
// Output is Option<T>
where
    T: Reflect + FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static,
    U: Reflect + FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static,
{
    None // Always returns None
}

/// Extension trait providing combinator methods for types implementing [`SignalVec`] and [`Clone`].
pub trait SignalVecExt: SignalVec + Clone {
    // Use consolidated SignalVec trait
    /// Creates a new `SignalVec` which maps the items within the output diffs of this `SignalVec`
    /// using the given Bevy system `F: IntoSystem<In<Self::Item>, Option<U>, M>`.
    ///
    /// The provided system `F` is run for each relevant item within the incoming `VecDiff<Self::Item>`
    /// (e.g., for `Push`, `InsertAt`, `UpdateAt`, `Replace`). If the system `F` returns `None` for an item,
    /// that item is effectively filtered out from the resulting `VecDiff<U>`. The structure of the diff
    /// (like `RemoveAt`, `Move`, `Pop`, `Clear`) is preserved.
    ///
    /// The system `F` must be `Clone`, `Send`, `Sync`, and `'static`.
    fn map<U, M, F>(self, system: F) -> MapVec<Self, U>
    // F is IntoSystem
    where
        Self: Sized,
        <Self as SignalVec>::Item:
            Reflect + FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static,
        U: Reflect + FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static,
        F: IntoSystem<In<<Self as SignalVec>::Item>, U, M>
            // F takes In<T>, returns Option<U>
            + Send
            + Sync
            + Clone
            + 'static,
        M: Send + Sync + 'static;

    // TODO: Add other combinators like filter, len, etc.

    /// Registers a system that runs for each batch of `VecDiff`s emitted by this signal.
    ///
    /// The provided system `F` takes `In<Vec<VecDiff<Self::Item>>>` and returns `()`.
    /// This method consumes the signal stream at this point; no further signals are propagated.
    ///
    /// Returns a [`ForEachVec`] node representing this terminal operation.
    /// Call `.register(world)` on the result to activate the chain and get a [`SignalHandle`].
    fn for_each<M, F>(self, system: F) -> ForEachVec<Self>
    where
        Self: Sized,
        <Self as SignalVec>::Item:
            Reflect + FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static,
        F: IntoSystem<In<Vec<VecDiff<Self::Item>>>, Option<()>, M> + Send + Sync + Clone + 'static,
        M: Send + Sync + 'static;

    /// Registers all the systems defined in this `SignalVec` chain into the Bevy `World`.
    ///
    /// Returns a [`SignalHandle`] for potential cleanup.
    fn register(&self, world: &mut World) -> SignalHandle;
}

impl<T> SignalVecExt for T
where
    T: SignalVec + Clone, // Use consolidated SignalVec trait
{
    fn map<U, M, F>(self, system: F) -> MapVec<Self, U>
    // F is IntoSystem
    where
        <T as SignalVec>::Item:
            Reflect + FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static,
        U: Reflect + FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static,
        F: IntoSystem<In<<T as SignalVec>::Item>, U, M> // F takes In<T>, returns Option<U>
            + Send
            + Sync
            + Clone
            + 'static,
        M: Send + Sync + 'static,
    {
        // Clone the user's system for registration
        let f_clone = system.clone();

        let register_fn = Arc::new(
            move |world: &mut World, prev_last_id_entity: UntypedSystemId| -> UntypedSystemId {
                // 1. Register the user's system F to get its SystemId and Entity.
                let user_system_id = world.register_system(f_clone.clone());
                let user_entity = user_system_id.entity(); // Get the entity associated with F

                // 2. Register the terminator system.
                //    Input is Vec<VecDiff<U>>, Output is Option<T::Item>.
                let terminator_id = register_signal::<Vec<VecDiff<U>>, <T as SignalVec>::Item, _>(
                    world,
                    terminator_system::<<T as SignalVec>::Item, U>,
                );
                let terminator_entity = terminator_id.entity();

                // 3. Define and register the wrapper system.
                //    Input is Vec<VecDiff<T::Item>>, Output is Option<Vec<VecDiff<U>>>.
                let wrapper_system =
                    move |In(diff_batch): In<Vec<VecDiff<<T as SignalVec>::Item>>>, // Use SignalVec::Item
                          world: &mut World|
                          -> Option<Vec<VecDiff<U>>> {
                        // ... (internal logic of wrapper system unchanged) ...
                        let mut output_batch: Vec<VecDiff<U>> = Vec::new();

                        for diff_t in diff_batch {
                            let maybe_diff_u: Option<VecDiff<U>> = match diff_t {
                                VecDiff::Replace { values } => {
                                    let mapped_values: Vec<U> = values
                                        .into_iter()
                                        .filter_map(|v| {
                                            world.run_system_with_input(user_system_id, v).ok()
                                        })
                                        .collect();
                                    Some(VecDiff::Replace {
                                        values: mapped_values,
                                    })
                                }
                                VecDiff::InsertAt { index, value } => world
                                    .run_system_with_input(user_system_id, value)
                                    .ok()
                                    .map(|mapped_value| VecDiff::InsertAt {
                                        index,
                                        value: mapped_value,
                                    }),
                                VecDiff::UpdateAt { index, value } => world
                                    .run_system_with_input(user_system_id, value)
                                    .ok()
                                    .map(|mapped_value| VecDiff::UpdateAt {
                                        index,
                                        value: mapped_value,
                                    }),
                                VecDiff::Push { value } => world
                                    .run_system_with_input(user_system_id, value)
                                    .ok()
                                    .map(|mapped_value| VecDiff::Push {
                                        value: mapped_value,
                                    }),
                                // Pass through structural variants
                                VecDiff::RemoveAt { index } => Some(VecDiff::RemoveAt { index }),
                                VecDiff::Move {
                                    old_index,
                                    new_index,
                                } => Some(VecDiff::Move {
                                    old_index,
                                    new_index,
                                }),
                                VecDiff::Pop => Some(VecDiff::Pop),
                                VecDiff::Clear => Some(VecDiff::Clear),
                            };

                            if let Some(diff_u) = maybe_diff_u {
                                output_batch.push(diff_u);
                            }
                        }

                        if output_batch.is_empty() {
                            None
                        } else {
                            Some(output_batch)
                        }
                    };

                // Register wrapper: Input Vec<VecDiff<T>>, Output Vec<VecDiff<U>>
                let wrapper_id = register_signal(world, wrapper_system);
                let wrapper_entity = wrapper_id.entity();

                // 4. Pipe the signals together in the desired chain.
                pipe_signal(world, prev_last_id_entity, wrapper_entity); // Previous -> Wrapper
                pipe_signal(world, wrapper_entity, terminator_entity); // Wrapper -> Terminator
                pipe_signal(world, terminator_entity, user_entity); // Terminator -> User System F's Entity

                // Return the wrapper entity as it represents the output of the map operation.
                wrapper_entity
            },
        );

        MapVec {
            prev_signal: self,
            register_fn,
            _marker: PhantomData,
        }
    }

    fn for_each<M, F>(self, system: F) -> ForEachVec<Self>
    where
        <T as SignalVec>::Item:
            Reflect + FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static,
        F: IntoSystem<In<Vec<VecDiff<Self::Item>>>, Option<()>, M> + Send + Sync + Clone + 'static,
        M: Send + Sync + 'static,
    {
        let system_clone = system.clone();

        let register_fn = Arc::new(
            move |world: &mut World, prev_last_id_entity: UntypedSystemId| -> UntypedSystemId {
                // Register the user's system. Input: Vec<VecDiff<T::Item>>, Output: ()
                // register_signal wraps it to return Option<()>
                let final_system_id = register_signal::<Vec<VecDiff<<T as SignalVec>::Item>>, (), M>(
                    world, // Pass world to register_signal
                    system_clone.clone(),
                );
                let final_system_entity = final_system_id.entity();

                // Pipe the previous signal's output into the user's system.
                pipe_signal(world, prev_last_id_entity, final_system_entity); // Pass world to pipe_signal

                // Return the entity of the registered user system.
                final_system_entity
            },
        );

        // Return the ForEachVec struct, not a SignalHandle
        ForEachVec {
            prev_signal: self,
            register_fn,
        }
    }

    fn register(&self, world: &mut World) -> SignalHandle {
        // Directly call the register method from the consolidated SignalVec trait
        <T as SignalVec>::register(self, world)
    }
}

//-------------------------------------------------------------------------------------------------
// Lock Guards
//-------------------------------------------------------------------------------------------------

/// A read guard for `MutableVec`, providing immutable access to the underlying `Vec`.
/// The guard holds the read lock, ensuring safe access.
pub struct MutableVecReadGuard<'a, T>
where
    T: Reflect + FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static + Clone,
{
    guard: RwLockReadGuard<'a, MutableVecState<T>>,
}

impl<'a, T> Deref for MutableVecReadGuard<'a, T>
where
    T: Reflect + FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static + Clone,
{
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        // SAFETY: The guard ensures the data is valid for the lifetime 'a.
        &self.guard.vec
    }
}

/// A write guard for `MutableVec`, providing mutable access to the underlying `Vec`.
/// The guard holds the write lock. Changes made through this guard automatically queue
/// the corresponding `VecDiff`s.
pub struct MutableVecWriteGuard<'a, T>
where
    T: Reflect + FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static + Clone,
{
    guard: RwLockWriteGuard<'a, MutableVecState<T>>,
}

// We cannot directly implement DerefMut to &mut Vec<T> because we need to intercept
// mutations to queue diffs. Instead, we provide explicit methods.

impl<'a, T> MutableVecWriteGuard<'a, T>
where
    T: Reflect + FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static + Clone,
{
    /// Returns a mutable reference to the element at `index`, or `None` if out of bounds.
    /// Note: This method does *not* automatically queue an `UpdateAt` diff.
    /// Use `set` for that. This is for complex mutations where a single `UpdateAt`
    /// isn't appropriate, or when you intend to call other methods like `remove` afterwards.
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.guard.vec.get_mut(index)
    }

    /// Returns mutable slices covering the whole vector.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.guard.vec.as_mut_slice()
    }

    // --- Methods that queue diffs ---

    /// Pushes a value to the end of the vector and queues a `VecDiff::Push`.
    pub fn push(&mut self, value: T) {
        self.guard.vec.push(value.clone());
        self.guard.pending_diffs.push(VecDiff::Push { value });
    }

    /// Removes the last element from the vector and returns it, or `None` if it is empty.
    /// Queues a `VecDiff::Pop` if an element was removed.
    pub fn pop(&mut self) -> Option<T> {
        let result = self.guard.vec.pop();
        if result.is_some() {
            self.guard.pending_diffs.push(VecDiff::Pop);
        }
        result
    }

    /// Inserts an element at `index` within the vector, shifting all elements after it to the right.
    /// Queues a `VecDiff::InsertAt`.
    /// # Panics
    /// Panics if `index > len`.
    pub fn insert(&mut self, index: usize, value: T) {
        self.guard.vec.insert(index, value.clone());
        self.guard
            .pending_diffs
            .push(VecDiff::InsertAt { index, value });
    }

    /// Removes and returns the element at `index` within the vector, shifting all elements after it to the left.
    /// Queues a `VecDiff::RemoveAt`.
    /// # Panics
    /// Panics if `index` is out of bounds.
    pub fn remove(&mut self, index: usize) -> T {
        let value = self.guard.vec.remove(index);
        self.guard.pending_diffs.push(VecDiff::RemoveAt { index });
        value
    }

    /// Removes all elements from the vector.
    /// Queues a `VecDiff::Clear` if the vector was not empty.
    pub fn clear(&mut self) {
        if !self.guard.vec.is_empty() {
            self.guard.vec.clear();
            self.guard.pending_diffs.push(VecDiff::Clear);
        }
    }

    /// Updates the element at `index` with a new `value`.
    /// Queues a `VecDiff::UpdateAt`.
    /// # Panics
    /// Panics if `index` is out of bounds.
    pub fn set(&mut self, index: usize, value: T) {
        let len = self.guard.vec.len();
        if index < len {
            self.guard.vec[index] = value.clone();
            self.guard
                .pending_diffs
                .push(VecDiff::UpdateAt { index, value });
        } else {
            panic!(
                "MutableVecWriteGuard::set: index {} out of bounds for len {}",
                index, len
            );
        }
    }

    /// Moves an item from `old_index` to `new_index`.
    /// Queues a `VecDiff::Move` if the indices are different and valid.
    /// # Panics
    /// Panics if `old_index` or `new_index` are out of bounds.
    pub fn move_item(&mut self, old_index: usize, new_index: usize) {
        let len = self.guard.vec.len();
        if old_index >= len || new_index >= len {
            panic!(
                "MutableVecWriteGuard::move_item: index out of bounds (len: {}, old: {}, new: {})",
                len, old_index, new_index
            );
        }

        if old_index != new_index {
            let value = self.guard.vec.remove(old_index);
            self.guard.vec.insert(new_index, value);
            self.guard.pending_diffs.push(VecDiff::Move {
                old_index,
                new_index,
            });
        }
    }

    /// Replaces the entire contents of the vector with the provided `values`.
    /// Queues a `VecDiff::Replace`.
    pub fn replace(&mut self, values: Vec<T>) {
        self.guard.vec = values.clone();
        self.guard.pending_diffs.push(VecDiff::Replace { values });
    }

    // --- Provide immutable access via Deref ---
    // This allows reading the state even with a write guard.
}

impl<'a, T> Deref for MutableVecWriteGuard<'a, T>
where
    T: Reflect + FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static + Clone,
{
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        // SAFETY: The guard ensures the data is valid for the lifetime 'a.
        &self.guard.vec
    }
}

//-------------------------------------------------------------------------------------------------
// MutableVec - Updated to use Guards
//-------------------------------------------------------------------------------------------------

/// Internal state for `MutableVec`, allowing `MutableVec` to be `Clone`.
#[derive(Debug)]
struct MutableVecState<T>
where
    T: Reflect + FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static + Clone,
{
    vec: Vec<T>,
    pending_diffs: Vec<VecDiff<T>>,
    listeners: Vec<UntypedSystemId>,
}

/// A mutable vector that tracks changes as `VecDiff`s and sends them as a batch on `flush`.
/// This struct is `Clone`able, sharing the underlying state.
#[derive(Debug, Clone)]
pub struct MutableVec<T>
where
    T: Reflect + FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static + Clone,
{
    state: Arc<RwLock<MutableVecState<T>>>,
}

#[derive(Component)]
struct QueuedVecDiffs<T: FromReflect + GetTypeRegistration + Typed>(Vec<VecDiff<T>>);

impl<T, A: Clone + FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static> From<T>
    for MutableVec<A>
where
    Vec<A>: From<T>,
{
    #[inline]
    fn from(values: T) -> Self {
        MutableVec {
            state: Arc::new(RwLock::new(MutableVecState {
                vec: values.into(),
                pending_diffs: Vec::new(),
                listeners: Vec::new(),
            })),
        }
    }
}

impl<T> MutableVec<T>
where
    T: Reflect + FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static + Clone, // Removed PartialEq
{
    /// Creates a new, empty `MutableVec`.
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(MutableVecState {
                vec: Vec::new(),
                pending_diffs: Vec::new(),
                listeners: Vec::new(),
            })),
        }
    }

    /// Creates a new `MutableVec` initialized with the given values.
    pub fn with_values(values: Vec<T>) -> Self {
        Self {
            state: Arc::new(RwLock::new(MutableVecState {
                pending_diffs: Vec::new(), // Start with empty diffs
                vec: values,
                listeners: Vec::new(),
            })),
        }
    }

    /// Acquires a read lock, returning a guard that provides immutable access.
    #[inline]
    pub fn read(&self) -> MutableVecReadGuard<'_, T> {
        MutableVecReadGuard {
            guard: self.state.read().unwrap(),
        }
    }

    /// Acquires a write lock, returning a guard that provides mutable access
    /// and automatically queues diffs for modifications.
    #[inline]
    pub fn write(&self) -> MutableVecWriteGuard<'_, T> {
        MutableVecWriteGuard {
            guard: self.state.write().unwrap(),
        }
    }

    // --- Convenience methods (delegate to guards) ---

    /// Returns the number of elements in the vector (acquires read lock).
    pub fn len(&self) -> usize {
        self.read().len()
    }

    /// Returns `true` if the vector contains no elements (acquires read lock).
    pub fn is_empty(&self) -> bool {
        self.read().is_empty()
    }

    /// Returns a clone of the element at `index`, or `None` if out of bounds (acquires read lock).
    pub fn get(&self, index: usize) -> Option<T> {
        self.read().get(index).cloned()
    }

    /// Pushes a value to the end of the vector (acquires write lock).
    pub fn push(&self, value: T) {
        self.write().push(value);
    }

    /// Removes the last element from the vector and returns it (acquires write lock).
    pub fn pop(&self) -> Option<T> {
        self.write().pop()
    }

    /// Inserts an element at `index` (acquires write lock).
    pub fn insert(&self, index: usize, value: T) {
        self.write().insert(index, value);
    }

    /// Removes and returns the element at `index` (acquires write lock).
    pub fn remove(&self, index: usize) -> T {
        self.write().remove(index)
    }

    /// Removes all elements from the vector (acquires write lock).
    pub fn clear(&self) {
        self.write().clear();
    }

    /// Updates the element at `index` with a new `value` (acquires write lock).
    pub fn set(&self, index: usize, value: T) {
        self.write().set(index, value);
    }

    /// Moves an item from `old_index` to `new_index` (acquires write lock).
    pub fn move_item(&self, old_index: usize, new_index: usize) {
        self.write().move_item(old_index, new_index);
    }

    /// Replaces the entire contents of the vector (acquires write lock).
    pub fn replace(&self, values: Vec<T>) {
        self.write().replace(values);
    }

    // --- Signal-related methods (unchanged for now, still need direct lock access) ---

    /// Creates a [`SourceVec<T>`] signal linked to this `MutableVec`.
    pub fn signal_vec(&self, world: &mut World) -> SourceVec<T> {
        // ... (Implementation remains the same, using self.state.read/write directly) ...
        // Register the identity batch system.
        // Input is Vec<VecDiff<T>>, Output is Option<Vec<VecDiff<T>>>.
        let source_system_id_lock = Arc::new(OnceLock::new());
        let source_system_id = register_signal::<(), Vec<VecDiff<T>>, _>(world, {
            let source_system_id_lock = source_system_id_lock.clone();
            move |_: In<()>, world: &mut World| {
                world
                    .get_entity_mut(source_system_id_lock.get().copied().unwrap())
                    .ok()
                    .and_then(|mut entity: EntityWorldMut<'_>| {
                        entity
                            .get_mut::<QueuedVecDiffs<T>>()
                            .map(|mut x| x.0.drain(..).collect())
                    })
            }
        });
        let entity = source_system_id.entity();

        // Initialize with Replace diff containing current state
        let initial_values = self.state.read().unwrap().vec.clone(); // Read lock to get initial values
        world
            .entity_mut(entity)
            .insert(QueuedVecDiffs(vec![VecDiff::Replace {
                values: initial_values,
            }]));
        source_system_id_lock.set(entity).unwrap();

        // Mark this system as a root for propagation
        crate::tree::mark_signal_root(world, entity);

        // Add listener to the shared state
        self.state.write().unwrap().listeners.push(entity); // Write lock to add listener

        SourceVec {
            entity,
            _marker: PhantomData,
        }
    }

    /// Sends any pending `VecDiff`s accumulated since the last flush to the signal system.
    pub fn flush(&self, world: &mut World) {
        // ... (Implementation remains the same, using self.state.write directly) ...
        let mut state = self.state.write().unwrap(); // Acquire write lock once
        if !state.pending_diffs.is_empty() {
            for &listener in &state.listeners {
                if let Ok(mut entity) = world.get_entity_mut(listener) {
                    if let Some(mut queued_diffs) = entity.get_mut::<QueuedVecDiffs<T>>() {
                        queued_diffs.0.extend(state.pending_diffs.clone()); // Clone diffs to extend
                    }
                }
            }
            state.pending_diffs.clear();
        }
    }

    // Remove the vec() method, use read() guard instead.
    // pub fn vec(&self) -> Vec<T> { ... }
}

// Remove Deref/DerefMut implementations for MutableVec itself
// impl<T> Deref for MutableVec<T> ...
// impl<T> DerefMut for MutableVec<T> ...
