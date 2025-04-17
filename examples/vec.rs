use bevy::prelude::*;
use jonmo::prelude::*; // Use jonmo prelude
use rand::prelude::*;
// Removed unused Duration import

// Component holding the independent value for each entity
#[derive(Component, Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(Component)]
struct Value(i32);

// Resource to hold the MutableVec tracking entities in the "column"
#[derive(Resource)]
struct ColumnEntities(MutableVec<Entity>);

// System to periodically increment the Value of all entities that have it
fn increment_all_values(mut query: Query<&mut Value>, time: Res<Time>, mut timer: Local<f32>) {
    *timer += time.delta_secs(); // Use delta_secs()
    // Increment roughly twice per second
    if *timer >= 0.5 {
        *timer = 0.0;
        for mut value in query.iter_mut() {
            value.0 += 1;
            // println!("Incremented value for entity: {:?}", entity); // Too noisy
        }
    }
}

// System to randomly add/remove entities from the MutableVec
fn randomly_modify_column(
    mut commands: Commands,
    mut column: ResMut<ColumnEntities>,
    time: Res<Time>,
    mut timer: Local<f32>,
) {
    *timer += time.delta_secs(); // Use delta_secs()
    // Modify roughly every 1.5 seconds
    if *timer < 1.5 {
        return;
    }
    *timer = 0.0;

    // Get thread-local RNG here
    let mut rng = rand::thread_rng();

    let mut modified = false;
    let current_entities = &mut column.0; // Access the inner MutableVec mutably

    // Decide whether to add or remove
    // Use public len() method
    if current_entities.len() < 2 || rng.gen_bool(0.6) {
        // Add a new entity
        let new_entity = commands.spawn(Value(0)).id();
        current_entities.push(new_entity);
        println!("Added Entity {:?} to column", new_entity);
        modified = true;
    // Use public is_empty() method
    } else if !current_entities.is_empty() {
        // Remove a random entity
        // Use public len() method
        let index_to_remove = rng.gen_range(0..current_entities.len());
        let entity_to_remove = current_entities.remove(index_to_remove);
        println!("Removed Entity {:?} from column", entity_to_remove);
        // Optionally despawn the entity if it should be completely gone
        // commands.entity(entity_to_remove).despawn();
        modified = true;
    }

    // Flush the MutableVec to send diffs if modified
    if modified {
        commands.queue(|world: &mut World| {
            if let Some(mut e) = world.remove_resource::<ColumnEntities>() {
                e.0.flush(world);
                world.insert_resource(e);
            }
        });
        // Use public len() method
        println!(
            "Flushed MutableVec. Current column size: {}",
            current_entities.len()
        );
    }
}

// Final consumer system that prints the VecDiff<String>
// Modified to return Option<VecDiff<()>> to satisfy map's trait bound
fn ui_column_updater(In(diff_opt): In<Option<VecDiff<String>>>) -> Option<VecDiff<()>> {
    if let Some(diff) = diff_opt {
        // In a real app, apply this diff to update the UI column display
        println!("UI Update Diff: {:?}", diff);
    }
    None // Indicate the end of this signal path for the frame
}

// Setup system to initialize resources and register the SignalVec chain
fn setup_signal_vec(world: &mut World) {
    println!("Setting up SignalVec...");

    // Initialize the MutableVec resource
    let mut column_entities = MutableVec::<Entity>::new();

    // Create the signal source from the MutableVec
    let source_signal = column_entities.signal_vec(world);

    // Insert the MutableVec as a resource AFTER creating the signal source
    world.insert_resource(ColumnEntities(column_entities));

    // Define the signal chain
    let signal_vec_chain = source_signal.map(|x: In<Entity>| {
        println!("here {:?}", x);
        Some(1)
    });
    // .map(ui_column_updater); // Final consumer system

    // Register the chain
    let _handle = signal_vec_chain.register(world); // Store handle if cleanup needed
    println!("SignalVec registered.");
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(JonmoPlugin) // Add Jonmo plugin
        .register_type::<Value>() // Register component type
        .add_systems(Startup, setup_signal_vec)
        .add_systems(
            Update,
            (
                // increment_all_values,
                // Removed .after() for now to fix build error
                randomly_modify_column,
            ),
        )
        .run();
}
