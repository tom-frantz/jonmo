use bevy_ecs::prelude::*; // Removed unused system::SystemId
use bevy_reflect::{FromReflect, GetTypeRegistration, Typed, prelude::*}; // Add reflection traits
use std::{
    fmt,
    marker::PhantomData,
    ops::{Deref, DerefMut},
    sync::{Arc, OnceLock},
}; // Added Deref, DerefMut

use crate::{
    signal::SignalHandle, // Use SignalHandle::new constructor
    tree::{UntypedSystemId, pipe_signal, register_signal}, // Removed mark_signal_root
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
pub trait SignalVec: Send + Sync + 'static {
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
    T: Reflect + FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static,
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
pub trait SignalVecExt: SignalVec + Clone { // Use consolidated SignalVec trait
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
        F: IntoSystem<In<<Self as SignalVec>::Item>, Option<U>, M>
            // F takes In<T>, returns Option<U>
            + Send
            + Sync
            + Clone
            + 'static,
        M: Send + Sync + 'static;

    // TODO: Add other combinators like filter, len, etc.

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
        F: IntoSystem<In<<T as SignalVec>::Item>, Option<U>, M> // F takes In<T>, returns Option<U>
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
                                    println!("replacing");
                                    let mapped_values: Vec<U> = values
                                        .into_iter()
                                        .filter_map(|v| {
                                            world
                                                .run_system_with_input(user_system_id, v)
                                                .ok()
                                                .flatten()
                                        })
                                        .collect();
                                    Some(VecDiff::Replace {
                                        values: mapped_values,
                                    })
                                }
                                VecDiff::InsertAt { index, value } => world
                                    .run_system_with_input(user_system_id, value)
                                    .ok()
                                    .flatten()
                                    .map(|mapped_value| VecDiff::InsertAt {
                                        index,
                                        value: mapped_value,
                                    }),
                                VecDiff::UpdateAt { index, value } => world
                                    .run_system_with_input(user_system_id, value)
                                    .ok()
                                    .flatten()
                                    .map(|mapped_value| VecDiff::UpdateAt {
                                        index,
                                        value: mapped_value,
                                    }),
                                VecDiff::Push { value } => {
                                    println!("pushing");
                                    world
                                        .run_system_with_input(user_system_id, value)
                                        .ok()
                                        .flatten()
                                        .map(|mapped_value| VecDiff::Push {
                                            value: mapped_value,
                                        })
                                }
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

    fn register(&self, world: &mut World) -> SignalHandle {
        // Directly call the register method from the consolidated SignalVec trait
        <T as SignalVec>::register(self, world)
    }
}

//-------------------------------------------------------------------------------------------------
// MutableVec - Reverted to pending_diffs and flush sending batch
//-------------------------------------------------------------------------------------------------

/// A mutable vector that tracks changes as `VecDiff`s and sends them as a batch on `flush`.
#[derive(Debug)]
pub struct MutableVec<T>
where
    T: Reflect + FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static + Clone, // Removed PartialEq
{
    vec: Vec<T>,
    pending_diffs: Vec<VecDiff<T>>, // Store pending changes
    listeners: Vec<UntypedSystemId>,
}

#[derive(Component)]
struct QueuedVecDiffs<T: FromReflect + GetTypeRegistration + Typed>(Vec<VecDiff<T>>);

impl<T> MutableVec<T>
where
    T: Reflect + FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static + Clone, // Removed PartialEq
{
    /// Creates a new, empty `MutableVec`.
    pub fn new() -> Self {
        Self {
            vec: Vec::new(),
            pending_diffs: Vec::new(),
            listeners: Vec::new(),
        }
    }

    /// Creates a new `MutableVec` initialized with the given values.
    pub fn with_values(values: Vec<T>) -> Self {
        Self {
            pending_diffs: Vec::new(),
            vec: values,
            listeners: Vec::new(),
        }
    }

    /// Creates a [`SourceVec<T>`] signal linked to this `MutableVec`.
    /// Registers an identity system to process batches of diffs sent by `flush`.
    pub fn signal_vec(&mut self, world: &mut World) -> SourceVec<T> {
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
        world
            .entity_mut(entity)
            .insert(QueuedVecDiffs(vec![VecDiff::Replace {
                values: self.vec.clone(),
            }]));
        source_system_id_lock.set(entity).unwrap();

        // Mark this system as a root for propagation
        crate::tree::mark_signal_root(world, entity);

        self.listeners.push(entity);

        SourceVec {
            entity,
            _marker: PhantomData,
        }
    }

    /// Sends any pending `VecDiff`s accumulated since the last flush to the signal system.
    pub fn flush(&mut self, world: &mut World) {
        if !self.pending_diffs.is_empty() {
            for &listener in &self.listeners {
                if let Ok(mut entity) = world.get_entity_mut(listener) {
                    if let Some(mut queued_diffs) = entity.get_mut::<QueuedVecDiffs<T>>() {
                        println!("Flushing pending diffs");
                        queued_diffs.0.extend(self.pending_diffs.clone());
                    }
                }
            }
            self.pending_diffs.clear();
        }
    }

    // --- Modification Methods (Add diffs to pending_diffs) ---

    /// Pushes a value to the end of the vector and queues a `VecDiff::Push`.
    pub fn push(&mut self, value: T) {
        self.vec.push(value.clone());
        self.pending_diffs.push(VecDiff::Push { value });
    }

    /// Removes the last element from the vector and returns it, or `None` if it is empty.
    /// Queues a `VecDiff::Pop` if an element was removed.
    pub fn pop(&mut self) -> Option<T> {
        let result = self.vec.pop();
        if result.is_some() {
            self.pending_diffs.push(VecDiff::Pop);
        }
        result
    }

    /// Inserts an element at `index` within the vector, shifting all elements after it to the right.
    /// Queues a `VecDiff::InsertAt`.
    /// # Panics
    /// Panics if `index > len`.
    pub fn insert(&mut self, index: usize, value: T) {
        self.vec.insert(index, value.clone());
        self.pending_diffs.push(VecDiff::InsertAt { index, value });
    }

    /// Removes and returns the element at `index` within the vector, shifting all elements after it to the left.
    /// Queues a `VecDiff::RemoveAt`.
    /// # Panics
    /// Panics if `index` is out of bounds.
    pub fn remove(&mut self, index: usize) -> T {
        let value = self.vec.remove(index);
        self.pending_diffs.push(VecDiff::RemoveAt { index });
        value
    }

    /// Removes all elements from the vector.
    /// Queues a `VecDiff::Clear` if the vector was not empty.
    pub fn clear(&mut self) {
        if !self.vec.is_empty() {
            self.vec.clear();
            self.pending_diffs.push(VecDiff::Clear);
        }
    }

    /// Updates the element at `index` with a new `value`.
    /// Queues a `VecDiff::UpdateAt`.
    /// # Panics
    /// Panics if `index` is out of bounds.
    pub fn set(&mut self, index: usize, value: T) {
        let len = self.vec.len();
        if index < len {
            self.vec[index] = value.clone();
            self.pending_diffs.push(VecDiff::UpdateAt { index, value });
        } else {
            panic!(
                "MutableVec::set: index {} out of bounds for len {}",
                index, len
            );
        }
    }

    /// Moves an item from `old_index` to `new_index`.
    /// Queues a `VecDiff::Move` if the indices are different and valid.
    /// # Panics
    /// Panics if `old_index` or `new_index` are out of bounds.
    pub fn move_item(&mut self, old_index: usize, new_index: usize) {
        let len = self.vec.len();
        if old_index >= len || new_index >= len {
            panic!(
                "MutableVec::move_item: index out of bounds (len: {}, old: {}, new: {})",
                len, old_index, new_index
            );
        }

        if old_index != new_index {
            let value = self.vec.remove(old_index);
            self.vec.insert(new_index, value);
            self.pending_diffs.push(VecDiff::Move {
                old_index,
                new_index,
            });
        }
    }

    /// Replaces the entire contents of the vector with the provided `values`.
    /// Queues a `VecDiff::Replace`.
    pub fn replace(&mut self, values: Vec<T>) {
        self.vec = values.clone();
        self.pending_diffs.push(VecDiff::Replace { values });
    }

    // ... Accessor Methods (len, is_empty, vec - unchanged) ...
}

// Implement Deref to allow accessing slice methods directly
impl<T> Deref for MutableVec<T>
where
    T: Reflect + FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static + Clone,
{
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.vec
    }
}

// Implement DerefMut to allow accessing mutable slice methods directly
impl<T> DerefMut for MutableVec<T>
where
    T: Reflect + FromReflect + GetTypeRegistration + Typed + Send + Sync + 'static + Clone,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.vec
    }
}

// ... Default impl for MutableVec ...
