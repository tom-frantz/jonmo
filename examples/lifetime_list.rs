mod utils;
use utils::*;

use bevy::prelude::*;
use jonmo::prelude::*;

fn main() {
    let mut app = App::new();
    let colors = MutableVec::from([random_color(), random_color()]);
    app.add_plugins(DefaultPlugins)
        .add_plugins(JonmoPlugin)
        .insert_resource(Colors(colors.clone()))
        .add_systems(
            PostStartup,
            (
                move |world: &mut World| {
                    ui_root(colors.signal_vec(world)).spawn(world);
                },
                camera,
            ),
        )
        .add_systems(
            Update,
            (live.run_if(any_with_component::<Lifetime>), hotkeys),
        )
        .run();
}

#[derive(Resource, Clone)]
struct Colors(MutableVec<Color>);

fn flush_colors(world: &mut World) {
    if let Some(colors) = world.remove_resource::<Colors>() {
        colors.0.flush(world);
        world.insert_resource(colors);
    }
}

#[derive(Component, Default, Clone, Reflect)]
struct Lifetime(f32);

fn ui_root(colors: impl SignalVec<Item = Color>) -> NodeBuilder {
    NodeBuilder::from(Node {
        height: Val::Percent(100.0),
        width: Val::Percent(100.0),
        flex_direction: FlexDirection::Column,
        align_items: AlignItems::Center,
        justify_content: JustifyContent::Center,
        row_gap: Val::Px(10.0),
        ..default()
    })
    .children_signal_vec(colors.map(|In(color)| item(color)))
}

fn item(color: Color) -> NodeBuilder {
    NodeBuilder::from((
        Node {
            height: Val::Px(40.0),
            width: Val::Px(200.0),
            padding: UiRect::all(Val::Px(5.0)),
            align_items: AlignItems::Center,
            ..default()
        },
        BackgroundColor(color),
    ))
    .child(
        NodeBuilder::from((
            Node {
                height: Val::Percent(100.),
                width: Val::Percent(100.),
                ..default()
            },
            TextColor(Color::BLACK),
            TextLayout::new_with_justify(JustifyText::Center),
            Lifetime::default(),
        ))
        .component_signal_from_component(|signal| {
            signal
                .map(|In(Lifetime(lifetime))| Some(lifetime.round()))
                .map(dedupe)
                .map(|In(lifetime)| Some(Some(Text::new(format!("lifetime: {}", lifetime)))))
        }),
    )
}

fn camera(mut commands: Commands) {
    commands.spawn(Camera2d::default());
}

fn live(mut lifetimes: Query<&mut Lifetime>, time: Res<Time>) {
    for mut lifetime in lifetimes.iter_mut() {
        lifetime.0 += time.delta_secs();
    }
}

fn hotkeys(keys: Res<ButtonInput<KeyCode>>, colors: ResMut<Colors>, mut commands: Commands) {
    if keys.just_pressed(KeyCode::Equal) {
        colors.0.push(random_color());
        commands.run_system_cached(flush_colors);
    } else if keys.just_pressed(KeyCode::Minus) {
        colors.0.pop();
        commands.run_system_cached(flush_colors);
    }
}
