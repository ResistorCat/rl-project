"""
Training command implementation.
"""

import logging
from poke_env.player import RandomPlayer
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor

from environment.wrapper import PokeEnvSinglesWrapper
from utils.logging_config import configure_poke_env_logging
from utils.types import RLModel
from utils.output_utils import get_output_dir
from utils.plot_utils import plot_training_learning_curve


def train_command(
    model_type: RLModel = RLModel.PPO,
    restart_server: bool = False,
    dev_mode: bool = False,
    initialize_func=None,
    cleanup_func=None,
    server=None,
    no_docker=False,
):
    """
    Train the model with the given name.

    Args:
        model_type: The type of model to train
        restart_server: Whether to restart the server before training
        dev_mode: Whether to run in development mode (shorter training)
        initialize_func: Function to initialize the environment
        cleanup_func: Function to clean up resources
        server: Server instance for restart operations
        no_docker: Whether running in no-docker mode
    """
    logger = logging.getLogger("Training")

    try:
        # Initialize environment
        if initialize_func:
            initialize_func(no_docker=no_docker)

        # Handle server restart if requested
        if restart_server and server is not None:
            logger.info("🔄 Restarting server as requested...")
            server.restart()

        # Create training environment
        logger.info("🎮 Setting up training environment...")
        env = PokeEnvSinglesWrapper(
            battle_format="gen9randombattle",
            log_level=30,  # WARNING level to reduce verbosity
            start_challenging=True,
            strict=False,
        )
        random_player = RandomPlayer()
        train_env = env.get_wrapped_env(opponent=random_player)

        # Set output dir
        output_dir = get_output_dir(task_type="train", model_type=model_type)
        model_path = output_dir / f"{model_type.value}_model.zip"
        monitor_path = output_dir / f"{model_type.value}_monitor.csv"

        # Monitor training
        train_env = Monitor(
            train_env,
            filename=str(monitor_path),
            allow_early_resets=True,
            override_existing=True,
        )

        # Configure PokeEnv logging to reduce noise
        configure_poke_env_logging()
        logger.info("🔇 Configured PokeEnv logging to reduce verbosity")

        # Determine training duration based on mode
        if dev_mode:
            logger.info("🛠️ Running in DEVELOPMENT mode (faster training for testing)")
            total_timesteps = 5000
        else:
            logger.info("🏭 Running in PRODUCTION mode (full training)")
            total_timesteps = 100_000

        logger.info(f"🚀 Training model: {model_type.value}")

        model = None
        if model_type == RLModel.PPO:
            model = PPO(
                "MlpPolicy",
                train_env,
                verbose=1,
                device="cpu",  # Force CPU usage
            )
        elif model_type == RLModel.DQN:
            model = DQN(
                "MlpPolicy",
                train_env,
                verbose=1,
            )
        else:
            logger.error(f"❌ Unknown model type: {model_type}")
            raise ValueError(f"Unsupported model type: {model_type}")

        if model:
            # Train the model
            model.learn(total_timesteps=total_timesteps)
            logger.info("✅ Training session completed!")

            # Save model
            model.save(model_path)
            logger.info(f"💾 Model saved to: {model_path}")

            # Generate learning curve plot
            try:
                logger.info("📊 Generating learning curve plot...")
                save_path = output_dir / f"{model_type.value}_learning_curve.png"
                plot_training_learning_curve(
                    model_type=model_type,
                    monitor_path=monitor_path,
                    save_path=save_path,
                )
                logger.info(f"📈 Learning curve plot saved to: {save_path}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to generate learning curve plot: {e}")

        # Close the environment
        train_env.close()
        env.close()

    except KeyboardInterrupt:
        logger.warning("🛑 Training interrupted by user")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise
    finally:
        # Clean up resources
        if cleanup_func and not no_docker:
            cleanup_func()
