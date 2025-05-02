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

use bevy_app::prelude::*;
use bevy_ecs::{prelude::*, system::SystemState};

mod builder;
mod signal;
mod signal_vec;
mod tree;
pub mod utils;
pub use builder::EntityBuilder;

use bevy_reflect::PartialReflect;
// Publicly export items from modules
pub use signal::*;
pub use signal_vec::{MapVec, MutableVec, SignalVec, SignalVecExt, SourceVec, VecDiff};
pub use tree::{pipe_signal, register_signal}; // Export SignalVec types

use tree::{Downstream, SignalSystem, SystemRunner, Upstream};
use utils::SSs;

fn clone_downstream(downstream: &Downstream) -> Vec<SignalSystem> {
    downstream.iter().cloned().collect()
}

fn process_signals_helper(
    world: &mut World,
    signals: impl IntoIterator<Item = SignalSystem>,
    input: Box<dyn PartialReflect>,
) {
    for signal in signals {
        if let Some(runner) = world
            .get_entity(*signal)
            .ok()
            .and_then(|entity| entity.get::<SystemRunner>().cloned())
        {
            if let Some(output) = runner.run(world, input.clone_value()) {
                if let Some(downstream) = world.get::<Downstream>(*signal).map(clone_downstream) {
                    process_signals_helper(world, downstream, output);
                }
            }
        }
    }
}

/// System that drives signal propagation by calling [`SignalPropagator::execute`].
/// Added to the `Update` schedule by the [`JonmoPlugin`]. This system runs once per frame.
/// It temporarily removes the [`SignalPropagator`] resource to allow mutable access to the `World`
/// during system execution within the propagator.
pub(crate) fn process_signals(world: &mut World) {
    let mut orphaned_parent_signals = SystemState::<
        Query<Entity, (With<SystemRunner>, Without<Upstream>, With<Downstream>)>,
    >::new(world);
    let orphaned_parent_signals = orphaned_parent_signals.get(world);
    let orphaned_parent_signals = orphaned_parent_signals
        .iter()
        .map(SignalSystem)
        .collect::<Vec<_>>();
    process_signals_helper(world, orphaned_parent_signals, Box::new(()));
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
        app.add_systems(Last, (process_signals, flush_cleanup_signals).chain());
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
        self as jonmo, JonmoPlugin,
        builder::*,
        signal::{Combine, Map, Signal, SignalBuilder, SignalExt, SignalHandle, Source},
        signal_vec::{MapVec, MutableVec, SignalVec, SignalVecExt, SourceVec, VecDiff},
    };
}

/// A generic identity system that takes an input `T` and returns `Some(T)`.
/// Useful for signal graph nodes that just pass through data.
///
/// Typically used with `In<T>` for Bevy system parameters.
pub fn identity<T: SSs>(In(input): In<T>) -> Option<T> {
    Some(input)
}
