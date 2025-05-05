# Tetris

## Description

Custom implementation of tetris with basic decoupled UIs.

## Prerequisites

Install [uv](https://docs.astral.sh/uv/getting-started/installation/)

## Usage

Running

```sh
uv run tetris
```

shows the available CLI commands. Using one of the commands with `--help` shows their available subcommands and options.

### Examples

#### Play

Play a number of tetris games in parallel, with configurable controllers (like keyboard, gamepad, automated bot).

Play a basic game of tetris alone (using keyboard controls (WASD, vim keys, or arrow keys)):

```sh
uv run tetris play
```

Have 2 players (one on keyboard (WASD), one on gamepad) and one bot compete:

```sh
uv run tetris play --controller wasd --controller gamepad --controller bot
```

Note that this requires the extra dependency `gamepad` to be installed using `uv sync --extra gamepad`.

Let 5 bots compete one the same seed, sped up 10x, on a custom 40x20 board:

```sh
uv run tetris play -n5 -cbot --seed same --fps 600 --board-size 40x20
```

#### Fill Space

The animation that plays on startup shown when playing tetris is an algorithm that fills a given 2-dimensional space with tetrominoes, and colors than with 4 different colors such that no neighboring tetrominoes have the same color. It can be executed on its own, with user-defined parameters.

Continuously fill and color random spaces, with random parameters:

```sh
uv run tetris fill-space fuzz-test
```

Fill a space with the size of the current terminal window with tetrominoes, but don't 4-color them (every single tetromino is colored differently):

```sh
uv run tetris fill-space no-color
```

First fill a space with the size of the current terminal window with tetrominoes, then 4-color them as a separate step:

```sh
uv run tetris fill-space color subsequent
```

Fill a space with the size of the current terminal window with tetrominoes, and concurrently color the so far placed tetrominoes:  
(Note that this is less performant than `subsequent`)

```sh
uv run tetris fill-space color concurrent
```

#### Train

Bots use a heuristic to decide where to place tetris pieces. This heuristic can be trained to become more optimal using a genetic algorithm.

Train the tetris heuristic bot from scratch (i.e. from default heuristic parameters which have been found through training, so not truly from scratch), using a genetic algorithm and a population size of 100:

```sh
uv run tetris train from-scratch --population-size 100
```

Continue training from the latest saved checkpoint at the default checkpoint directory:

```sh
uv run tetris train from-checkpoint
```

#### Evaluate

Evaluate a bot with a given heuristic on a number of tetris games with different seeds in parallel, to get a good idea of how well it performs. This produces an evaluation report that contains mean, median, max, and min scores, and more info.

Evaluate the default heuristic (previously trained to perform well) on 50 games:

```sh
uv run tetris evaluate explicit
```

Evaluate the top 15 best performing bots from a training checkpoint:

```sh
uv run tetris evaluate from-train-checkpoint path/to/checkpoint.pkl --top-k 15
```

Re-Evaluate the top 5 best performing bots from a previous evaluation report file, on a different board size:

```sh
uv run tetris evaluate --board-size 30x15 from-evaluation-report path/to/report.csv --top-k 5
```

## Pre-commit hooks

Pre-commit hooks can be installed using

```sh
uvx pre-commit install
```

manually run using

```sh
uvx pre-commit run --all-files
```

and kept up-to-date using

```sh
uvx pre-commit autoupdate
```

## Testing

```sh
uv run pytest
```
