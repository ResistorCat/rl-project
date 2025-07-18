"""
Evaluation command implementation.
"""

import logging
import numpy as np
from poke_env.player import RandomPlayer, MaxBasePowerPlayer
from stable_baselines3 import PPO, DQN

from environment.wrapper import PokeEnvSinglesWrapper, DQNPlayer
from utils.types import RLModel, RLPlayer
from utils.evaluation_utils import EvaluationResults
from utils.logging_config import configure_poke_env_logging
from utils.output_utils import get_output_dir
from tqdm.rich import trange

def evaluate_command(
    model_type: RLModel = RLModel.PPO,
    initialize_func=None,
    cleanup_func=None,
    no_docker=False,
    opponents: list[RLPlayer] = [RLPlayer.RANDOM],
    num_battles: int = 1000,
    name: str | None = None,
):
    """
    Evaluate the model and generate training progress plots.
    """
    logger = logging.getLogger("Evaluation")

    try:
        # Initialize environment
        if initialize_func:
            initialize_func(no_docker=no_docker)

        # Load the latest trained model of the type specified
        logger.info(f"🔄 Loading model: {name} (Type: {model_type})")
        model_path = (
            get_output_dir(task_type="train", model_type=model_type)
            / f"{name if name else model_type.value}_model.zip"
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

        results = EvaluationResults(model_type=model_type, name=name)

        # Configure PokeEnv logging to reduce noise
        configure_poke_env_logging()
        logger.info("🔇 Configured PokeEnv logging to reduce verbosity")
        for opponent in opponents:
            if opponent == RLPlayer.RANDOM:
                player = RandomPlayer(log_level=30)
                opponent_name = "Random"
            elif opponent == RLPlayer.MAX:
                player = MaxBasePowerPlayer(log_level=30)
                opponent_name = "MaxBasePower"
            elif opponent == RLPlayer.DQN_RANDOM:
                # Load all DQN models trained
                opponent_model_path = (
                    get_output_dir(task_type="train", model_type=RLModel.DQN)
                    / "random_model.zip"
                )
                if not opponent_model_path.exists():
                    logger.error(
                        "❌ DQN model not found. Please train the DQN model first."
                    )
                    continue
                player = DQNPlayer(
                    model=DQN.load(opponent_model_path, device="cpu")
                )
                opponent_name = "DQN (Random Train)"
            elif opponent == RLPlayer.DQN_MAX:
                # Load all DQN models trained
                opponent_model_path = (
                    get_output_dir(task_type="train", model_type=RLModel.DQN)
                    / "max_model.zip"
                )
                if not opponent_model_path.exists():
                    logger.error(
                        "❌ DQN model not found. Please train the DQN model first."
                    )
                    continue
                player = DQNPlayer(
                    model=DQN.load(opponent_model_path, device="cpu")
                )
                opponent_name = "DQN (Max Train)"
            else:
                logger.error(f"❌ Unsupported opponent: {opponent}")
                continue

            # Create evaluation environment
            logger.info("🎮 Setting up evaluation environment...")
            env = PokeEnvSinglesWrapper(
                battle_format="gen9randombattle",
                log_level=30,  # WARNING level to reduce verbosity
                start_challenging=True,
                strict=False,
            )
            eval_env = env.get_wrapped_env(opponent=player)
            
            # Run evaluation battles
            logger.info(
                f"⚔️ Running {num_battles} evaluation battles against {opponent.value}..."
            )
            battle_rewards = []
            battle_results = []  # True for win, False for loss
            battle_steps = []  # Track number of steps per battle

            for battle_num in trange(1, num_battles + 1):
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
                    obs, reward, terminated, truncated, info = eval_env.step(
                        action_value
                    )
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
                opponent_name=opponent_name,
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
