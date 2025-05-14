# Tetris

## Overview

A custom implementation of Tetris featuring an ANSI-powered terminal UI and flexible controller support. Includes commands for playing, training and evaluating Tetris bots, and visualizing a custom space filling and coloring algorithm used in the startup animation of the Tetris UI.
Developed and mainly tested on Windows. Windows Terminal was used for development and is recommended, but most modern terminals should work fine (e.g. VS Code integrated terminal works mostly fine).

## Prerequisites

- [Install uv](https://docs.astral.sh/uv/getting-started/installation/) (for dependency management and running commands)

## Usage

To see available CLI commands, run:

```sh
uv run tetris
```

> *Note: when running for the first time, you might have to wait a bit for dependencies to be installed*

Use `--help` with any command to view its subcommands and options.

### Examples

#### Play

Run one or more Tetris games in parallel, each with configurable controllers (keyboard, gamepad, or automated bot).

- **Single-player (keyboard controls: WASD, vim keys, or arrows):**

  ```sh
  uv run tetris play
  ```

- **Multiplayer (keyboard, gamepad, and bot):**

  ```sh
  uv run tetris play --controller wasd --controller gamepad --controller bot
  ```

  > *Note: Gamepad support requires the `gamepad` extra. Install with:*
  >
  > ```sh
  > uv sync --extra gamepad
  > ```

- **Bot competition (5 bots, same seed, fast-forward, custom board size, with Tetris99 rules (cleared lines are sent to an opponent), without sound):**

  ```sh
  uv run tetris play -n5 -cbot --seed same --fps 600 --board-size 40x20 --tetris99 --sounds off
  ```

#### Fill Space

Visualize an algorithm that fills a 2D space with tetrominoes and colors them using four colors so that no adjacent pieces share a color. This animation is shown at startup but can also be run independently.

- **Fuzz test (random spaces and parameters):**

  ```sh
  uv run tetris fill-space fuzz-test
  ```

- **No 4-coloring (each tetromino gets a unique color):**

  ```sh
  uv run tetris fill-space no-color
  ```

- **Sequential coloring (fill, then color):**

  ```sh
  uv run tetris fill-space color subsequent
  ```

- **Concurrent coloring (color as you fill; less performant):**

  ```sh
  uv run tetris fill-space color concurrent
  ```

#### Train

Train a bot's heuristic using a genetic algorithm to optimize its Tetris gameplay.

- **Start training from default parameters (population size 100):**

  ```sh
  uv run tetris train from-scratch --population-size 100
  ```

- **Continue training from the latest checkpoint:**

  ```sh
  uv run tetris train from-checkpoint
  ```

#### Evaluate

Assess a bot's performance by running it on multiple games with different seeds. Generates a report with statistics (mean, median, max, min scores, etc.).

- **Evaluate the default heuristic on 50 games:**

  ```sh
  uv run tetris evaluate explicit
  ```

- **Evaluate top 15 bots from a training checkpoint:**

  ```sh
  uv run tetris evaluate from-train-checkpoint path/to/checkpoint.pkl --top-k 15
  ```

- **Re-evaluate top 5 bots from a previous report on a different board size:**

  ```sh
  uv run tetris evaluate --board-size 30x15 from-evaluation-report path/to/report.csv --top-k 5
  ```

## Development

### Pre-commit Hooks

Install hooks:

```sh
uvx pre-commit install
```

Run hooks manually:

```sh
uvx pre-commit run --all-files
```

Update hooks:

```sh
uvx pre-commit autoupdate
```

### Testing

Run all tests:

```sh
uv run pytest
```

## Code structure

This section is meant to give a broad, non-exhaustive overview over the structure of the project.

- **[src/tetris/cli/](src/tetris/cli/)**  
  Implementation of the above described Tetris commandline interface, implemented using `click`.

- **[src/tetris/game_logic/](src/tetris/game_logic/)**  
  Scaffolding for the game loop: a `Runtime` class which contains a number of `Game` classes, which in turn contain `Rule`s and `Callback`s that are called every frame, and a `UI` that is used to draw the current game state every frame.

  - **[src/tetris/game_logic/components/](src/tetris/game_logic/components/)**  
    Core building blocks that the tetris game is made of (tetris pieces and the board they are placed on).

  - **[src/tetris/game_logic/interfaces/](src/tetris/game_logic/interfaces/)**  
    Interfaces (ABCs / Protocols) for classes used by the game loop scaffolding. Includes `Publisher` and `Subscriber` superclasses, which are the main way that different objects communicate.

  - **[src/tetris/game_logic/rules/](src/tetris/game_logic/rules/)**  
    Implementations of the `Rule` protocol. Rules are called every frame and define how the game works. They for example make pieces drop, clear full lines, and define how pieces move based on controller input. Also, message tuples that are passed between rules (and other objects) for communication.

- **[src/tetris/ui/](src/tetris/ui/)**  
  Implementations of the `UI` interface. Includes a CLI implementation powered by ANSI codes.

- **[src/tetris/ansi_extensions/](src/tetris/ansi_extensions/)**  
  Custom extension to the `ansi` package, adding more codes in the same style.

- **[src/tetris/clock/](src/tetris/clock/)**  
  Implementations of the `Clock` interface.

- **[src/tetris/controllers/](src/tetris/controllers/)**  
  Implementations of the `Controller` interface. Includes keyboard and gamepad controllers, as well as LLM and Heuristic-based bot controllers.

- **[src/tetris/genetic_algorithm.py](src/tetris/genetic_algorithm.py)**  
  Generic implementation of Genetic Algorithm.

- **[src/tetris/heuristic_gym/](src/tetris/heuristic_gym/)**  
  Code related to training (using Genetic Algorithm) and evaluating heuristic bot controllers.

- **[src/tetris/space_filling_coloring/](src/tetris/space_filling_coloring/)**  
  Code related to filling a space with tetrominoes and 4-coloring it, corresponding to the `fill-space` CLI command.

- **[tests/](tests/)**  
  Unit tests.

- **[logs/](logs/)**  
  Default location for log files.

- **[data/](data/)**  
  Default location for train checkpoint, evaluation report, and sound files.
