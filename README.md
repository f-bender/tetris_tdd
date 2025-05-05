# Tetris

## Overview

A custom implementation of Tetris featuring an ANSI-powered terminal UI and flexible controller support. Includes tools for playing, training and evaluating Tetris bots, and visualizing a custom space filling and coloring algorithm used in the startup animation of the Tetris UI.

## Prerequisites

- [Install uv](https://docs.astral.sh/uv/getting-started/installation/) (for dependency management and running commands)

## Usage

To see available CLI commands, run:

```sh
uv run tetris
```

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

- **Bot competition (5 bots, same seed, fast-forward, custom board size):**

  ```sh
  uv run tetris play -n5 -cbot --seed same --fps 600 --board-size 40x20
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
