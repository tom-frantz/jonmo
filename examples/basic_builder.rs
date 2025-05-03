use bevy::prelude::*;
use jonmo::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_plugins(DefaultPlugins)
        .add_plugins(JonmoPlugin)
        .add_systems(
            Startup,
            (
                |world: &mut World| {
                    ui_root().spawn(world);
                },
                camera,
            ),
        )
        .add_systems(Update, incr_value)
        .insert_resource(ValueTicker(Timer::from_seconds(1., TimerMode::Repeating)))
        .run();
}

#[derive(Resource, Deref, DerefMut)]
struct ValueTicker(Timer);

#[derive(Component, Reflect, Clone, Default, PartialEq)]
struct Value(i32);

fn ui_root() -> JonmoBuilder {
    JonmoBuilder::from(Node {
        justify_content: JustifyContent::Center,
        align_items: AlignItems::Center,
        height: Val::Percent(100.),
        width: Val::Percent(100.),
        ..default()
    })
    .child(
        JonmoBuilder::from(Node::default())
            .insert(Value(0))
            .component_signal_from_component(|signal| {
                signal.map(|In(value): In<Value>| Some(Text(value.0.to_string())))
            }),
    )
}

fn incr_value(mut ticker: ResMut<ValueTicker>, time: Res<Time>, mut values: Query<&mut Value>) {
    if ticker.tick(time.delta()).finished() {
        for mut value in values.iter_mut() {
            value.0 = value.0.wrapping_add(1);
        }
        ticker.reset();
    }
}

fn camera(mut commands: Commands) {
    commands.spawn(Camera2d);
}
