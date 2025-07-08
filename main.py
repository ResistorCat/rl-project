import typer
import logging
from environment.server import PokemonShowdownServer
from commands import train_command, evaluate_command
from utils.types import RLModel, RLPlayer
from utils.logging_config import setup_logging, configure_poke_env_logging
from utils.docker_utils import check_docker_availability


# Global server instance
server = None
NO_DOCKER = False

app = typer.Typer(
    help="CLI for training and evaluating Pokémon Showdown RL models. Use --dev flag for faster development testing.",
    add_completion=False,
)


@app.callback()
def main(
    no_docker: bool = typer.Option(
        False,
        "--no-docker",
        help="Run in manual mode without Docker (you'll need to start the server manually)",
    ),
):
    """
    Pokémon Showdown RL Training CLI
    """
    global NO_DOCKER
    NO_DOCKER = no_docker


def initialize(no_docker: bool = False):
    """
    Initialize the environment and server.

    Args:
        no_docker: If True, run in manual mode without Docker
    """
    logger = setup_logging()
    logger.info("Initializing the Pokémon RL environment...")

    global server

    if no_docker:
        logger.info("🔧 MANUAL MODE: Docker integration disabled")
        logger.warning(
            "You need to start the Pokémon Showdown server manually before continuing"
        )
        logger.info("Please ensure the server is running on the expected port")
        input("Press Enter to continue when the server is ready...")
        return

    # Check Docker availability
    logger.info("🐳 Checking Docker availability...")
    if not check_docker_availability():
        logger.error("❌ Docker is not available or not running on this system")
        logger.info("To run without Docker, use the --no-docker flag")

        # Offer manual mode as fallback
        response = (
            input("Would you like to continue in manual mode instead? (y/N): ")
            .strip()
            .lower()
        )
        if response in ["y", "yes"]:
            logger.info("🔧 Switching to manual mode...")
            logger.warning("You need to start the Pokémon Showdown server manually")
            input("Press Enter to continue when the server is ready...")
            return
        else:
            logger.error("Exiting. Please install Docker or use --no-docker flag")
            raise typer.Exit(code=1)

    logger.info("✅ Docker is available")

    # Initialize and start server
    try:
        server = PokemonShowdownServer()
        if not server.is_running():
            logger.info("🚀 Starting Pokémon Showdown server...")
            if not server.start():
                logger.error("❌ Failed to start the server")
                raise typer.Exit(code=1)
            logger.info("✅ Server started successfully")
        else:
            logger.info("✅ Server is already running")
    except Exception as e:
        logger.error(f"❌ An error occurred during server initialization: {e}")
        raise typer.Exit(code=1)


def cleanup():
    """Clean up resources, particularly stop the server if it was started."""
    global server
    logger = logging.getLogger("Cleanup")

    if server is not None:
        try:
            logger.info("🛑 Stopping Pokémon Showdown server...")

            # Configure logging to suppress websocket errors during shutdown
            configure_poke_env_logging()

            # Give a small delay for any active connections to finish
            import time

            time.sleep(1)

            server.stop()
            logger.info("✅ Server stopped successfully")
        except Exception as e:
            logger.warning(f"⚠️ Error stopping server: {e}")


@app.command()
def train(
    model: RLModel = RLModel.PPO,
    restart_server: bool = False,
    dev: bool = typer.Option(
        False,
        "--dev",
        help="Run in development mode (faster training with 5000 timesteps for testing)",
    ),
    opponent: RLPlayer = typer.Option(
        RLPlayer.RANDOM,
        "--opponent",
        help="Opponent player type for training (default: RANDOM)",
    ),
    timesteps: int = typer.Option(
        100_000, "--timesteps", help="Total timesteps for training (default: 100000)"
    ),
    name: str = typer.Option(
        None,
        "--name",
        help="Custom name for the saved model (default: None, uses model type)",
    ),
):
    """
    Train the model with the given name.
    """
    # Delegate to the train command implementation
    train_command(
        model_type=model,
        restart_server=restart_server,
        dev_mode=dev,
        initialize_func=initialize,
        cleanup_func=cleanup,
        server=server,
        no_docker=NO_DOCKER,
        opponent=opponent,
        total_timesteps=timesteps,
        name=name,
    )


@app.command()
def evaluate(
    model: RLModel = RLModel.PPO,
    opponents: list[RLPlayer] = typer.Option(
        [RLPlayer.RANDOM, RLPlayer.MAX, RLPlayer.DQN],
        "--opponents",
        help="List of opponents to evaluate against (default: all available players)",
    ),
    battles: int = typer.Option(
        100,
        "--battles",
        help="Number of battles to run for evaluation (default: 100)",
    ),
):
    """
    Evaluate the model and generate training progress plots.
    """
    # Delegate to the evaluate command implementation
    evaluate_command(
        model_type=model,
        initialize_func=initialize,
        cleanup_func=cleanup,
        no_docker=NO_DOCKER,
        opponents=opponents,
        num_battles=battles
    )


@app.command()
def clean(
    output: bool = typer.Option(
        True, "--output", help="Clean up output directories and files"
    ),
):
    """
    Clean up resources, particularly stop the server if it was started.
    """
    cleanup()
    if output:
        import shutil
        from utils.output_utils import get_output_dir

        logger = logging.getLogger("Cleanup")
        output_dir = get_output_dir()
        logger.info(f"🗑️ Cleaning up output directory: {output_dir}")

        # Remove the output directory if it exists
        if output_dir.exists():
            shutil.rmtree(output_dir)
            logger.info(f"✅ Output directory {output_dir} cleaned up")
        else:
            logger.info(
                f"ℹ️ Output directory {output_dir} does not exist, nothing to clean"
            )
    typer.echo("✅ Cleanup completed successfully")


if __name__ == "__main__":
    app()
