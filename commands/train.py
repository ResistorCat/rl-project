"""
Training command implementation.
"""

import logging
import time
from poke_env.player import RandomPlayer
from stable_baselines3 import PPO

from environment.wrapper import PokeEnvSinglesWrapper
from util import configure_poke_env_logging, RLModel


def train_command(
    model_type: RLModel = RLModel.PPO, 
    restart_server: bool = False,
    initialize_func=None,
    cleanup_func=None,
    server=None,
    no_docker=False
):
    """
    Train the model with the given name.
    
    Args:
        model_type: The type of model to train
        restart_server: Whether to restart the server before training
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
        train_env = env.get_training_env(opponent=random_player)
        
        # Configure PokeEnv logging to reduce noise
        configure_poke_env_logging()
        logger.info("🔇 Configured PokeEnv logging to reduce verbosity")

        logger.info(f"🚀 Training model: {model_type.value}")
        
        if model_type == RLModel.PPO:
            _train_ppo_model(train_env, logger)
        elif model_type == RLModel.DQN:
            # Placeholder for DQN training logic
            logger.warning("⚠️ DQN training is not implemented yet.")
        else:
            logger.error(f"❌ Unknown model type: {model_type}")
            raise ValueError(f"Unsupported model type: {model_type}")
        
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


def _train_ppo_model(train_env, logger):
    """Train a PPO model."""
    logger.info("🖥️ Using CPU device for training")
    
    # Wait a moment for environment to stabilize
    time.sleep(2)
    
    model_ppo = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        device="cpu",  # Force CPU usage
    )
    logger.info("🚀 Starting PPO training...")
    model_ppo.learn(total_timesteps=3_000)  # Reduced for testing
    logger.info("✅ Training completed.")
    model_ppo.save("ppo_pokemon_model")
    logger.info("💾 Model saved as 'ppo_pokemon_model'.")
