use bevy::prelude::*;
use jonmo::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_plugins(DefaultPlugins)
        .add_plugins(JonmoPlugin) // Use the JonmoPlugin struct
        // Register types needed for reflection
        .register_type::<Value>()
        // Add setup systems
        .add_systems(Startup, (setup_camera, setup_ui).chain()) // Use separate camera setup and declarative UI setup
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

// Declarative UI setup using the Signal builder
fn setup_ui(world: &mut World) {
    // Spawn entities directly within this system
    let entity = world.spawn((Node::default(), Value(0))).id();
    let text = world.spawn_empty().id();
    world.entity_mut(entity).add_child(text);

    // --- Signal Chain Setup (Declarative) ---

    // 1. Define the signal chain using the builder pattern.
    let signal = SignalBuilder::from_component::<Value>(entity) // Use Signals::from_component
        .map(dedupe)
        .map(move |In(value): In<Value>, mut commands: Commands| {
            // Map the changed Value to a system that updates text
            // Update text by inserting the Text component
            commands.entity(text).insert(Text(value.0.to_string()));
            TERMINATE // Signal propagation stops after this system runs.
        });

    // signal.combine_with(signal.clone());

    // 2. Register the entire chain with the world.
    // No need for disambiguation now, as SignalExt::register just defines the signature
    // and the implementation comes from SignalBuilderInternal::register.
    let _handle = signal.register(world); // Reverted to simple call
    // The handle can be stored if cleanup is needed later: handle.cleanup(world);
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
