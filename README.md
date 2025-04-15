# jonmo [জন্ম](https://translate.google.com/?sl=bn&tl=en&text=%E0%A6%9C%E0%A6%A8%E0%A7%8D%E0%A6%AE&op=translate)

[![Crates.io Version](https://img.shields.io/crates/v/jonmo?style=for-the-badge)](https://crates.io/crates/jonmo)
[![Docs.rs](https://img.shields.io/docsrs/jonmo?style=for-the-badge)](https://docs.rs/jonmo)
[![Following released Bevy versions](https://img.shields.io/badge/Bevy%20tracking-released%20version-lightblue?style=for-the-badge)](https://bevyengine.org/learn/quick-start/plugin-development/#main-branch-tracking)

```text
in bengali, jonmo means "birth"
```

jonmo provides a declarative way to define reactive signal chains in Bevy. Signals originate from sources (like component changes, resource changes, or specific entities) and can be transformed (`map`), combined (`combine_with`), or filtered (`dedupe`).

Internally, jonmo builds and maintains a dependency graph of Bevy systems. Each frame, the `JonmoPlugin` traverses this graph starting from the root systems. It pipes the output (`Some(O)`) of a parent system as the input (`In<O>`) to its children. This traversal continues down each branch until a system returns `None` (often represented by the [`TERMINATE`](https://docs.rs/jonmo/latest/jonmo/constant.TERMINATE.html) constant), which halts propagation along that specific path for the current frame.

## usage

jonmo allows you to build signal chains declaratively using the `Signal` builder.

### declarative api

Start a chain using `Signal::from_*` methods, chain transformations with `.map()`, and activate the chain with `.register()`.

```rust
use bevy::prelude::*;
use jonmo::*;

#[derive(Component, Reflect, Clone, Default, PartialEq)]
#[reflect(Component)]
struct Value(i32);

fn setup_ui(world: &mut World) {
    // Assume 'entity' and 'text_entity' are created elsewhere
    let entity = world.spawn(Value(0)).id();
    let text_entity = world.spawn_empty().id();

    // Define the signal chain
    let signal_chain = Signal::from_component::<Value>(entity) // React to Value changes on 'entity'
        .map(dedupe) // Only propagate if the value actually changed
        .map(move |In(value): In<Value>, mut commands: Commands| { // Update text when value changes
            commands.entity(text_entity).insert(Text(value.0.to_string()));
            TERMINATE // End the signal chain here for this update
        });

    // Register the chain to activate it
    let handle = signal_chain.register(world);
    // 'handle' can be used later to clean up the signal chain: handle.cleanup(world);
}

fn main() {
    let mut app = App::new();
    app.add_plugins(DefaultPlugins)
        .add_plugins(JonmoPlugin)
        .register_type::<Value>()
        .add_systems(Startup, setup_ui);
    // ... add systems to change Value ...
    app.run();
}
```

See the [declarative example](https://github.com/databasedav/jonmo/blob/main/examples/declarative.rs) for a runnable version.

### imperative api

For more control or integration with existing systems, you can use the lower-level imperative functions:

-   `register_signal`: Registers a single system node.
-   `mark_signal_root`: Marks a system as a starting point for propagation.
-   `pipe_signal`: Connects the output of one system to the input of another.
-   `combine_signal`: Creates the necessary nodes to combine two signal branches.

```rust
use bevy::prelude::*;
use jonmo::*; // Import imperative functions too

#[derive(Component, Reflect, Clone, Default, PartialEq)]
#[reflect(Component)]
struct Value(i32);

fn setup_ui_imperative(world: &mut World) {
    // Assume 'entity' and 'text_entity' are created elsewhere
    let entity = world.spawn(Value(0)).id();
    let text_entity = world.spawn_empty().id();

    // 1. Create root source
    let system1_id = register_signal(world, entity_root(entity));
    mark_signal_root(world, system1_id.entity());

    // 2. Create component change listener
    let system2_id = register_signal(world, |In(entity): In<Entity>, values: Query<&Value, Changed<Value>>| {
        values.get(entity).ok().cloned()
    });
    pipe_signal(world, system1_id.entity(), system2_id.entity());

    // 3. Create text update system
    let system3_id = register_signal(world, move |In(value): In<Value>, mut commands: Commands| {
        commands.entity(text_entity).insert(Text(value.0.to_string()));
        TERMINATE
    });
    pipe_signal(world, system2_id.entity(), system3_id.entity());
}
```

See the [imperative example](https://github.com/databasedav/jonmo/blob/main/examples/imperative.rs) for a runnable version.

## Bevy compatibility

|bevy|jonmo|
|-|-|

## license
All code in this repository is dual-licensed under either:

- MIT License ([LICENSE-MIT](https://github.com/databasedav/jonmo/blob/main/LICENSE-MIT) or <http://opensource.org/licenses/MIT>)
- Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/databasedav/jonmo/blob/main/LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)

at your option.

### your contributions
Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.


