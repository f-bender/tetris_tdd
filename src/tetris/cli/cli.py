import click

from tetris.cli.evaluate import evaluate
from tetris.cli.fill_space import fill_space
from tetris.cli.play import play
from tetris.cli.train import train
from tetris.logging_config import configure_logging


@click.group()
def cli() -> None:
    """A command-line interface for the Tetris project."""
    configure_logging()


cli.add_command(play)
cli.add_command(train)
cli.add_command(evaluate)
cli.add_command(fill_space)
