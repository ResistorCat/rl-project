"""
Training command implementation.
"""

import logging
import time
from poke_env.player import RandomPlayer
from stable_baselines3 import PPO

from environment.wrapper import PokeEnvSinglesWrapper
from util import configure_poke_env_logging, RLModel, get_model_path


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
        train_env = env.get_training_env(opponent=random_player, enable_monitor=True)

        # Configure PokeEnv logging to reduce noise
        configure_poke_env_logging()
        logger.info("🔇 Configured PokeEnv logging to reduce verbosity")

        # Log dev mode status
        if dev_mode:
            logger.info("🛠️ Running in DEVELOPMENT mode (faster training for testing)")
        else:
            logger.info("🏭 Running in PRODUCTION mode (full training)")

        logger.info(f"🚀 Training model: {model_type.value}")

        model_path = None
        if model_type == RLModel.PPO:
            model_path = _train_ppo_model(train_env, logger, dev_mode=dev_mode)
        elif model_type == RLModel.DQN:
            # Placeholder for DQN training logic
            logger.warning("⚠️ DQN training is not implemented yet.")
        else:
            logger.error(f"❌ Unknown model type: {model_type}")
            raise ValueError(f"Unsupported model type: {model_type}")

        if model_path:
            logger.info(f"✅ Training session completed! Model saved to: {model_path}")

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


def _train_ppo_model(train_env, logger, dev_mode: bool = False):
    """
    Train a PPO model.
    
    Args:
        train_env: The training environment
        logger: Logger instance
        dev_mode: Whether to run in development mode (shorter training)
    """
    logger.info("🖥️ Using CPU device for training")

    # Wait a moment for environment to stabilize
    time.sleep(2)

    model_ppo = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        device="cpu",  # Force CPU usage
    )
    
    # Determine training duration based on mode
    if dev_mode:
        total_timesteps = 5000
        logger.info(f"🚀 Starting PPO training (DEV MODE: {total_timesteps:,} timesteps)...")
    else:
        total_timesteps = 100_000
        logger.info(f"🚀 Starting PPO training (PRODUCTION: {total_timesteps:,} timesteps)...")
    
    model_ppo.learn(total_timesteps=total_timesteps)
    logger.info("✅ Training completed.")

    # Save model to organized directory structure
    model_path = get_model_path(RLModel.PPO, include_timestamp=True)
    model_ppo.save(str(model_path))
    logger.info(f"💾 Model saved to: {model_path}")

    return model_path
