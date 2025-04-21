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
use bevy_ecs::prelude::*;
// Declare modules
mod node_builder;
mod signal;
mod signal_vec;
mod tree; // Add signal_vec module
// mod mutable_vec; // Remove mutable_vec module

use prelude::RemoveSignalHandles;
// Publicly export items from modules
pub use signal::*;
pub use signal_vec::{
    MapVec, MutableVec, SignalVec, SignalVecBuilder, SignalVecExt, SourceVec, VecDiff,
};
pub use tree::{mark_signal_root, pipe_signal, register_signal}; // Export SignalVec types

use tree::{
    MarkSignalRoot, PipeSignal, SignalPropagator, consume_mark_signal_root, consume_pipe_signal,
    remove_signal_handles,
};

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
        app.init_resource::<SignalPropagator>()
            .add_event::<MarkSignalRoot>()
            .add_event::<PipeSignal>()
            .add_event::<RemoveSignalHandles>()
            .add_systems(
                First,
                (
                    consume_mark_signal_root.run_if(on_event::<MarkSignalRoot>),
                    consume_pipe_signal.run_if(on_event::<PipeSignal>),
                )
                    .run_if(resource_exists::<SignalPropagator>),
            )
            .add_systems(
                Last,
                (
                    remove_signal_handles.run_if(on_event::<RemoveSignalHandles>),
                    process_signals,
                )
                    .chain(),
            );
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
        JonmoPlugin,
        node_builder::*,
        signal::{Combine, Map, Signal, SignalBuilder, SignalExt, SignalHandle, Source},
        signal_vec::{
            MapVec, MutableVec, SignalVec, SignalVecBuilder, SignalVecExt, SourceVec, VecDiff,
        }, // Add SignalVec to prelude
        tree::{TERMINATE, dedupe, entity_root},
    };
    // Note: SignalBuilderInternal is intentionally excluded
    // Note: Imperative functions like register_signal, pipe_signal, mark_signal_root are also excluded from prelude
}

/// A generic identity system that takes an input `T` and returns `Some(T)`.
/// Useful for signal graph nodes that just pass through data.
///
/// Typically used with `In<T>` for Bevy system parameters.
pub fn identity<T: Send + Sync + 'static>(In(input): In<T>) -> Option<T> {
    Some(input)
}
