"""
Training command implementation.
"""

import logging
import time
from poke_env.player import RandomPlayer, MaxBasePowerPlayer
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor

from environment.wrapper import PokeEnvSinglesWrapper, DQNPlayer
from utils.logging_config import configure_poke_env_logging
from utils.types import RLModel, RLPlayer
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
    opponent: RLPlayer = RLPlayer.RANDOM,
    total_timesteps: int = 100_000,
    name: str | None = None,
):
    """
    Train the model with the given name.
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
        if opponent == RLPlayer.RANDOM:
            player = RandomPlayer(
                battle_format="gen9randombattle",
                log_level=30,  # WARNING level to reduce verbosity
            )
        elif opponent == RLPlayer.MAX:
            player = MaxBasePowerPlayer(
                battle_format="gen9randombattle",
                log_level=30,  # WARNING level to reduce verbosity
            )
        elif opponent == RLPlayer.DQN_RANDOM:
            # Check if DQN is trained
            opponent_model_path = (
                get_output_dir(task_type="train", model_type=RLModel.DQN)
                / "random_model.zip"
            )
            if not opponent_model_path.exists():
                raise FileNotFoundError(
                    f"❌ DQN model not found at {opponent_model_path}. "
                    "Please train the DQN model first."
                )
            player = DQNPlayer(model=DQN.load(opponent_model_path, device="cpu"))
        elif opponent == RLPlayer.DQN_MAX:
            # Check if DQN is trained
            opponent_model_path = (
                get_output_dir(task_type="train", model_type=RLModel.DQN)
                / "max_model.zip"
            )
            if not opponent_model_path.exists():
                raise FileNotFoundError(
                    f"❌ DQN model not found at {opponent_model_path}. "
                    "Please train the DQN model first."
                )
            player = DQNPlayer(model=DQN.load(opponent_model_path, device="cpu"))
        else:
            raise ValueError(f"Unsupported opponent type: {opponent}")
        train_env = env.get_wrapped_env(opponent=player)

        # Set output dir
        output_dir = get_output_dir(task_type="train", model_type=model_type)
        model_path = output_dir / f"{name if name else model_type.value}_model.zip"
        monitor_path = output_dir / f"{name if name else model_type.value}_monitor.csv"

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
            total_timesteps = 5_000
        else:
            logger.info("🏭 Running in PRODUCTION mode (full training)")

        logger.info(f"🚀 Training model {model_type.value} with name {name}")

        model = None
        if model_type == RLModel.PPO:
            model = PPO(
                "MlpPolicy",
                train_env,
                verbose=0,
                device="cpu",  # Force CPU usage
            )
        elif model_type == RLModel.DQN:
            model = DQN(
                "MlpPolicy",
                train_env,
                verbose=0,
            )
        else:
            logger.error(f"❌ Unknown model type: {model_type}")
            raise ValueError(f"Unsupported model type: {model_type}")

        if model:
            # Train the model
            time_start = time.time()
            logger.info(f"⏳ Starting training for {total_timesteps} timesteps...")
            model.learn(total_timesteps=total_timesteps, progress_bar=True)
            time_end = time.time()
            elapsed_time = time_end - time_start
            logger.info(f"⏱️ Training completed in {elapsed_time:.2f} seconds")

            # Save model
            model.save(model_path)
            logger.info(f"💾 Model saved to: {model_path}")

            # Generate learning curve plot
            try:
                logger.info("📊 Generating learning curve plot...")
                save_path = output_dir / f"{name if name else model_type.value}_learning_curve.png"
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
        logger.info("✅ Training completed successfully")

    except KeyboardInterrupt:
        logger.warning("🛑 Training interrupted by user")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise
    finally:
        # Clean up resources
        if cleanup_func and not no_docker:
            cleanup_func()
