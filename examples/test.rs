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

fn ui_root() -> EntityBuilder {
    EntityBuilder::from(Node {
        justify_content: JustifyContent::Center,
        align_items: AlignItems::Center,
        height: Val::Percent(100.),
        width: Val::Percent(100.),
        ..default()
    })
    .child(
        EntityBuilder::from(Node::default())
            .insert(Value(0))
            .component_signal_from_component(|signal| {
                signal.map(|In(value): In<Value>| Some(Text(value.0.to_string())))
            }),
    )
}

fn f(world: &mut World) {
    let signal = SignalBuilder::from_system(|_: In<()>| 1);
    signal.clone().register(world);  // 1 ref
    signal.clone().register(world);  // 2 ref
    let map = signal.map(|In(value): In<i32>, mut commands: Commands| {
        commands.spawn(Text::new(value.to_string()));
    });  // we don't want the root signal to be despawned when the others are, so we need to ref count it
    map.clone().register(world);  // 3 ref
    map.clone().register(world);  // 3 ref
    // cleaning up either of these should not remove the root signal, but cleaning up both should
    ;
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
