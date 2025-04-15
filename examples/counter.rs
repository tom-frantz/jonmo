//! Simple counter example using Bevy UI and jonmo signals.

use bevy::prelude::*;
use jonmo::*; // Import jonmo prelude

// --- Components ---

#[derive(Component, Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(Component)]
struct Counter(i32);

// Marker component for the text displaying the counter value
#[derive(Component)]
struct CounterText;

// Marker component to identify counter buttons and store their step value
#[derive(Component)]
struct CounterButton {
    step: i32,
}

// --- Constants ---

const NORMAL_BUTTON: Color = Color::rgb(0.8, 0.8, 0.8);
const HOVERED_BUTTON: Color = Color::rgb(0.7, 0.7, 0.7);
const PRESSED_BUTTON: Color = Color::rgb(0.6, 0.6, 0.6);

// --- Setup ---

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(JonmoPlugin)
        // Register component types
        .register_type::<Counter>()
        // Add setup systems
        .add_systems(Startup, (setup_camera, setup_ui).chain())
        // Add interaction and signal systems
        .add_systems(Update, button_interactions);

    app.run();
}

fn setup_camera(mut commands: Commands) {
    commands.spawn(Camera2dBundle::default());
}

// Spawn UI elements and register signals
fn setup_ui(mut commands: Commands) {
    // Spawn a dedicated entity to hold the counter state
    let counter_entity = commands.spawn(Counter(0)).id();

    // --- Spawn UI Tree ---
    let mut counter_text_entity = Entity::PLACEHOLDER; // Placeholder

    commands.spawn(NodeBundle { // Root node
        style: Style {
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
            column_gap: Val::Px(15.0),
            ..default()
        },
        ..default()
    }).with_children(|parent| {
        // "-" Button
        parent.spawn((
            ButtonBundle {
                style: Style {
                    width: Val::Px(45.0),
                    height: Val::Px(45.0),
                    justify_content: JustifyContent::Center,
                    align_items: AlignItems::Center,
                    ..default()
                },
                background_color: NORMAL_BUTTON.into(),
                ..default()
            },
            CounterButton { step: -1 },
        )).with_children(|parent| {
            parent.spawn(TextBundle::from_section("-", TextStyle { font_size: 25.0, ..default() }));
        });

        // Counter Display Text
        counter_text_entity = parent.spawn((
            TextBundle::from_section("0", TextStyle { font_size: 25.0, ..default() }),
            CounterText, // Mark this text entity
        )).id();

        // "+" Button
        parent.spawn((
            ButtonBundle {
                style: Style {
                    width: Val::Px(45.0),
                    height: Val::Px(45.0),
                    justify_content: JustifyContent::Center,
                    align_items: AlignItems::Center,
                    ..default()
                },
                background_color: NORMAL_BUTTON.into(),
                ..default()
            },
            CounterButton { step: 1 },
        )).with_children(|parent| {
            parent.spawn(TextBundle::from_section("+", TextStyle { font_size: 25.0, ..default() }));
        });
    });

    // --- Register Signals using Commands::add ---
    commands.add(move |world: &mut World| {
        // Signal chain to update the counter text display
        let update_text_signal = Signal::from_component::<Counter>(counter_entity)
            .map(dedupe) // Only update if the counter value actually changes
            .map(move |In(counter): In<Counter>, mut text_query: Query<&mut Text, With<CounterText>>| {
                if let Ok(mut text) = text_query.get_mut(counter_text_entity) {
                    // Update the text section
                    text.sections[0].value = counter.0.to_string();
                    println!("Counter Text Updated: {}", counter.0);
                }
                TERMINATE // End the chain here
            });

        // Register the signal chain
        let _handle = update_text_signal.register(world);
        // Could store handle if cleanup is needed: world.insert_resource(MyHandle(handle));
    });
}

// --- Interaction System ---

// System to handle button interactions (clicks and hover)
fn button_interactions(
    mut interaction_query: Query<
        (&Interaction, &mut BackgroundColor, &CounterButton),
        (Changed<Interaction>, With<Button>),
    >,
    mut counter_query: Query<&mut Counter>, // Query for the single Counter component
) {
    // Assume the counter entity is known or query for the single one
    // For simplicity, let's assume there's only one Counter entity.
    // In a real app, you might pass the entity ID via a resource or query differently.
    if let Ok(mut counter) = counter_query.get_single_mut() {
        for (interaction, mut color, button_data) in &mut interaction_query {
            match *interaction {
                Interaction::Pressed => {
                    *color = PRESSED_BUTTON.into();
                    // Modify the counter component directly
                    counter.0 += button_data.step;
                    println!("Button Pressed! New Counter: {}", counter.0);
                }
                Interaction::Hovered => {
                    *color = HOVERED_BUTTON.into();
                }
                Interaction::None => {
                    *color = NORMAL_BUTTON.into();
                }
            }
        }
    } else {
         // Handle case where Counter entity doesn't exist or isn't unique if needed
         // warn!("Could not find unique Counter entity");
    }
}
