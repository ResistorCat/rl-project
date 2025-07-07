"""
Evaluation command implementation.
"""

import logging
import numpy as np
from poke_env.player import RandomPlayer
from stable_baselines3 import PPO, DQN

from environment.wrapper import PokeEnvSinglesWrapper
from util.types import RLModel
from util.evaluation_utils import EvaluationResults
from util.logging_config import configure_poke_env_logging
from util.output_utils import get_output_dir


def evaluate_command(
    model_type: RLModel = RLModel.PPO,
    initialize_func=None,
    cleanup_func=None,
    no_docker=False,
):
    """
    Evaluate the model and generate training progress plots.

    Args:
        model: The type of model to evaluate
        model_path: Specific path to the model file (if None, uses latest)
        initialize_func: Function to initialize the environment
        cleanup_func: Function to clean up resources
        no_docker: Whether running in no-docker mode
    """
    logger = logging.getLogger("Evaluation")

    try:
        # Initialize environment
        if initialize_func:
            initialize_func(no_docker=no_docker)

        # Load the latest trained model of the type specified
        logger.info(f"🔄 Loading latest {model_type.value} model...")
        model_path = (
            get_output_dir(task_type="train", model_type=model_type)
            / f"{model_type.value}_model.zip"
        )
        if model_type == RLModel.PPO:
            trained_model = PPO.load(model_path, device="cpu")
        elif model_type == RLModel.DQN:
            trained_model = DQN.load(model_path)
        else:
            logger.error(
                f"❌ Model type {model_type.value} evaluation not implemented yet"
            )
            return
        logger.info("✅ Model loaded successfully")

        # Create evaluation environment
        logger.info("🎮 Setting up evaluation environment...")
        env = PokeEnvSinglesWrapper(
            battle_format="gen9randombattle",
            log_level=30,  # WARNING level to reduce verbosity
            start_challenging=True,
            strict=False,
        )
        random_player = RandomPlayer()
        eval_env = env.get_wrapped_env(opponent=random_player)

        # Configure PokeEnv logging to reduce noise
        configure_poke_env_logging()
        logger.info("🔇 Configured PokeEnv logging to reduce verbosity")
        # Run evaluation battles
        num_battles = 10
        logger.info(
            f"⚔️ Running {num_battles} evaluation battles against RandomPlayer..."
        )

        results = EvaluationResults(model_type=model_type)
        battle_rewards = []
        battle_results = []  # True for win, False for loss
        battle_steps = []  # Track number of steps per battle

        for battle_num in range(1, num_battles + 1):
            obs, info = eval_env.reset()
            done = False
            step_count = 0
            total_reward = 0

            while not done:
                # Use the trained model to predict actions
                action, _states = trained_model.predict(obs, deterministic=True)
                # Handle different action types (numpy array or scalar)
                if hasattr(action, "item"):
                    action_value = np.int64(action.item())
                else:
                    action_value = np.int64(action)
                obs, reward, terminated, truncated, info = eval_env.step(action_value)
                done = terminated or truncated
                step_count += 1
                total_reward += float(reward)

                # Prevent infinite loops
                if step_count > 1000:
                    logger.warning(
                        f"⚠️ Battle {battle_num} exceeded 1000 steps, ending battle"
                    )
                    break

            # Store battle results
            battle_rewards.append(total_reward)
            battle_steps.append(step_count)
            battle_won = total_reward > 0
            battle_results.append(battle_won)

        # Calculate overall statistics
        results.add_result(
            opponent_name="RandomPlayer",
            battles_won=sum(battle_results),
            total_battles=num_battles,
            mean_reward=np.mean(battle_rewards, dtype=np.float64),
            std_reward=np.std(battle_rewards, dtype=np.float64),
        )

        # Print and save
        results.print()
        results.save()

        # Close the environment
        eval_env.close()
        env.close()

    except KeyboardInterrupt:
        logger.warning("🛑 Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        raise
    finally:
        # Clean up resources
        if cleanup_func and not no_docker:
            cleanup_func()
