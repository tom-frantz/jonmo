use bevy::prelude::*;
use jonmo::{mark_signal_root, pipe_signal, prelude::*, register_signal};

fn main() {
    let mut app = App::new();
    app.add_plugins(DefaultPlugins)
        .add_plugins(JonmoPlugin) // Use the JonmoPlugin struct
        // Register types needed for reflection
        .register_type::<Value>()
        // Add setup systems
        .add_systems(Startup, (setup_camera, setup_ui).chain()) // Use separate camera setup and imperative UI setup
        // Add a system to test changes
        .add_systems(Update, change_values);

    app.run();
}

#[derive(Component, Reflect, Clone, Default, PartialEq)]
#[reflect(Component)] // Ensure reflection includes component info
struct Value(i32);

// System to spawn the camera
fn setup_camera(mut commands: Commands) {
    commands.spawn(Camera2d::default());
}

// Imperative UI setup using direct world access
fn setup_ui(world: &mut World) {
    // Spawn entities directly within this system
    let entity = world.spawn((Node::default(), Value(0))).id();
    let text = world.spawn_empty().id();
    world.entity_mut(entity).add_child(text);

    // --- Signal Chain Setup ---

    // 1. Create a root signal source that emits the target entity's ID.
    // `entity_root` creates a system that takes `In<()>` and returns `Some(entity)`.
    // `register_signal` registers this system with the Bevy world and gets its ID.
    let system1_id = register_signal(world, entity_root(entity));
    // `mark_signal_root` tells the SignalPropagator that this system should be run
    // automatically each update cycle, starting a potential signal cascade.
    mark_signal_root(world, system1_id.entity()); // Mark root explicitly

    // 2. Create a system that listens for changes to the `Value` component on the target entity.
    // This system takes the `Entity` ID from the previous signal (system1) as input (`In<Entity>`).
    // It queries for the `Value` component on that entity, but only if it has changed (`Changed<Value>`).
    // If the component exists and has changed, it returns `Some(cloned_value)`. Otherwise, `None`.
    // `register_signal` registers this system.
    let system2_id = register_signal(
        world,
        |In(entity): In<Entity>, values: Query<&Value, Changed<Value>>| {
            values.get(entity).ok().cloned()
        },
    );
    // `pipe_signal` connects the output of system1 (the Entity ID) to the input of system2.
    pipe_signal(world, system1_id.entity(), system2_id.entity());

    // 3. Create a system that takes the changed `Value` and updates the UI text.
    // This system takes the `Value` from the previous signal (system2) as input (`In<Value>`).
    // It uses `Commands` to insert/update the `Text` component on the `text` entity.
    // `move` is used to capture the `text` entity ID.
    // It returns `TERMINATE` (which is `None::<()>`) to stop the signal chain here.
    // `register_signal` registers this system.
    let system3_id = register_signal(
        world,
        move |In(value): In<Value>, mut commands: Commands| {
            // Update text by inserting the Text component
            commands.entity(text).insert(Text(value.0.to_string()));
            TERMINATE // Signal propagation stops after this system runs.
        },
    );
    // `pipe_signal` connects the output of system2 (the changed Value) to the input of system3.
    pipe_signal(world, system2_id.entity(), system3_id.entity());
}

// Add a system to modify values periodically for testing the signals
fn change_values(
    mut query: Query<&mut Value>,
    time: Res<Time>,
    mut timer: Local<f32>, // Use a local timer within the system
) {
    // Dereference time to access methods
    *timer += time.delta_secs();
    // Change values every second
    if *timer > 1.0 {
        *timer = 0.0; // Reset timer
        for mut val in query.iter_mut() {
            val.0 = val.0.wrapping_add(1); // Increment Value
            println!("Changed values: V={}", val.0);
        }
    }
}
