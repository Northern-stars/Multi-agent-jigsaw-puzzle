import os
import random
from typing import Dict, List

import numpy as np
import torch

from agent.mappo_dual_agent import DualBoardMAPPOAgent, MAPPOConfig
from env.dual_board_env import DualBoardEnv
from model_code.dual_board_mappo_model import DualBoardMAPPOModel
from utils.utils import plot_reward_curve, save_log


MODEL_NAME = "dual_board_mappo"
MODEL_PATH = os.path.join("model", f"{MODEL_NAME}.pth")
FILE_NAME = f"_{MODEL_NAME}"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_X_PATH = "dataset/train_img_48gap_33-001.npy"
TRAIN_Y_PATH = "dataset/train_label_48gap_33.npy"
VALID_X_PATH = "dataset/valid_img_48gap_33.npy"
VALID_Y_PATH = "dataset/valid_label_48gap_33.npy"

TOTAL_EPISODES = 5000
MAX_STEP = 200
LOAD_MODEL = False
SHOW_IMAGE = False
SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data() -> Dict[str, np.ndarray]:
    return {
        "train_x": np.load(TRAIN_X_PATH),
        "train_y": np.load(TRAIN_Y_PATH),
        "valid_x": np.load(VALID_X_PATH),
        "valid_y": np.load(VALID_Y_PATH),
    }


def save_checkpoint(agent: DualBoardMAPPOAgent) -> None:
    os.makedirs("model", exist_ok=True)
    torch.save(
        {
            "model": agent.model.state_dict(),
            "optimizer": agent.optimizer.state_dict(),
        },
        MODEL_PATH,
    )


def load_checkpoint(agent: DualBoardMAPPOAgent) -> None:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    agent.model.load_state_dict(checkpoint["model"])
    agent.optimizer.load_state_dict(checkpoint["optimizer"])


def run_training(env: DualBoardEnv, agent: DualBoardMAPPOAgent, epoch: int = TOTAL_EPISODES, load: bool = False) -> None:
    if load and os.path.exists(MODEL_PATH):
        load_checkpoint(agent)

    reward_record: List[List[float]] = []
    done_record: List[float] = []
    ownership_record: List[float] = []
    absolute_record: List[float] = []

    for episode in range(epoch):
        observations = env.reset()
        episode_reward = 0.0
        done = False
        step = 0
        info = env.get_metrics()

        while not done and step < MAX_STEP:
            actions, policy_info = agent.select_actions(observations, deterministic=False)
            next_observations, reward_list, done, info = env.step(actions)
            team_reward = float(sum(reward_list) / len(reward_list))
            agent.record_transition(policy_info, reward=team_reward, done=done)
            observations = next_observations
            episode_reward += team_reward
            step += 1

            if SHOW_IMAGE:
                env.show_image()

        update_info = agent.update(observations, done, show=True)
        reward_record.append([episode_reward])
        done_record.append(info["both_perfect"])
        ownership_record.append(info["ownership_accuracy"])
        absolute_record.append(info["overall_absolute"])

        print(
            f"Episode {episode + 1}/{epoch} | steps={step} | reward={episode_reward:.3f} | "
            f"ownership={info['ownership_accuracy']:.3f} | absolute={info['overall_absolute']:.3f} | "
            f"both_perfect={info['both_perfect']:.0f} | policy_loss={update_info['policy_loss']:.4f}"
        )

        save_checkpoint(agent)
        save_log("reward" + FILE_NAME, reward_record)
        save_log("done" + FILE_NAME, done_record)
        save_log("ownership" + FILE_NAME, ownership_record)
        save_log("absolute" + FILE_NAME, absolute_record)
        plot_reward_curve(reward_record, done_record, FILE_NAME)


if __name__ == "__main__":
    set_seed(SEED)
    data = load_data()

    env = DualBoardEnv(
        train_x=data["train_x"],
        train_y=data["train_y"],
        gamma=0.99,
        image_num=2,
        buffer_size=1,
        epsilon=0.1,
        epsilon_gamma=0.999,
        max_steps=MAX_STEP,
        cooldown_steps=5,
        training_mode=True,
        device=DEVICE,
    )

    model = DualBoardMAPPOModel(embed_dim=128, num_layers=3, num_heads=4, dropout=0.1).to(DEVICE)
    config = MAPPOConfig(
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        ppo_epochs=4,
        mini_batch_size=16,
        learning_rate=1e-4,
        intent_align_coef=0.1,
    )
    agent = DualBoardMAPPOAgent(model=model, env=env, config=config, device=DEVICE)

    run_training(env, agent, epoch=TOTAL_EPISODES, load=LOAD_MODEL)
