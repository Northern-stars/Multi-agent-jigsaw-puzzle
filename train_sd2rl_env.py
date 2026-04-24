import os
import random
from typing import Dict, List

import numpy as np
import torch

from agent.ppo_local_switcher import PPO_Local_switcher
from env.sd2rl_env import RewardConfig, SD2RLEnv
from model_code.local_switcher_model import Local_switcher_model
from utils.utils import model_fen_load, plot_reward_curve, save_log


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_X_PATH = "dataset/train_img_48gap_33-001.npy"
TRAIN_Y_PATH = "dataset/train_label_48gap_33.npy"
VALID_X_PATH = "dataset/valid_img_48gap_33.npy"
VALID_Y_PATH = "dataset/valid_label_48gap_33.npy"

MODEL_BACKBONE = "modulator"
ACTOR_MODEL_NAME = f"sd2rl_env_ppo_actor_{MODEL_BACKBONE}.pth"
CRITIC_MODEL_NAME = f"sd2rl_env_ppo_critic_{MODEL_BACKBONE}.pth"
ACTOR_MODEL_PATH = os.path.join("model", ACTOR_MODEL_NAME)
CRITIC_MODEL_PATH = os.path.join("model", CRITIC_MODEL_NAME)
FILE_NAME = f"_sd2rl_env_{MODEL_BACKBONE}"

TOTAL_EPISODES = 500
MAX_STEP = 80
TRAIN_PER_STEP = 10
SHOW_IMAGE = False
LOAD_MODEL = False
SEED = 42

GAMMA = 0.995
BATCH_SIZE = 32
MEMORY_SIZE = 2000

EPSILON = 0.5
EPSILON_MIN = 0.1
EPSILON_GAMMA = 0.998

SWAP_NUM_SCHEDULE = [
    (0, 5),
    (100, 5),
    (200, 8),
    (300, 8),
]


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


def get_swap_num(epoch: int) -> int:
    swap_num = SWAP_NUM_SCHEDULE[0][1]
    for threshold, value in SWAP_NUM_SCHEDULE:
        if epoch >= threshold:
            swap_num = value
    return swap_num


def build_env(train_x: np.ndarray, train_y: np.ndarray, epsilon: float) -> SD2RLEnv:
    return SD2RLEnv(
        train_x=train_x,
        train_y=train_y,
        epsilon=epsilon,
        epsilon_gamma=EPSILON_GAMMA,
        max_steps=MAX_STEP,
        initial_swap_num=5,
        device=DEVICE,
        reward_config=RewardConfig(pairwise=0.2, cate=0.8, done_reward=1000.0, step_penalty=-1.0),
        epochs=5,
    )


def build_switcher(env: SD2RLEnv) -> PPO_Local_switcher:
    model = Local_switcher_model(512, 512, 1024, 512, 1, model_name=MODEL_BACKBONE, dropout=0.1).to(DEVICE)
    model_fen_load(model, MODEL_BACKBONE)
    switcher = PPO_Local_switcher(
        model=model,
        memory_size=MEMORY_SIZE,
        gamma=GAMMA,
        batch_size=BATCH_SIZE,
        env=env,
        action_num=env.action_num,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        ppo_epochs=4,
    )
    if LOAD_MODEL:
        if os.path.exists(ACTOR_MODEL_PATH):
            switcher.model.load_state_dict(torch.load(ACTOR_MODEL_PATH, map_location=DEVICE))
        if os.path.exists(CRITIC_MODEL_PATH):
            switcher.critic_model.load_state_dict(torch.load(CRITIC_MODEL_PATH, map_location=DEVICE))
    return switcher


def train(env: SD2RLEnv, agent: PPO_Local_switcher, epoch_num: int = TOTAL_EPISODES) -> None:
    os.makedirs("model", exist_ok=True)

    reward_record: List[List[float]] = []
    done_record: List[float] = []
    cate_record: List[float] = []
    hori_record: List[float] = []
    vert_record: List[float] = []

    for epoch in range(epoch_num):
        swap_num = get_swap_num(epoch)
        state = env.reset(swap_num=swap_num)

        done = False
        step = 0
        episode_rewards: List[float] = []

        while not done and step < MAX_STEP:
            _, action = agent.act(state, 0)
            next_state, reward, done, info = env.step(action)

            agent.recording_memory(
                image_id=env.image_id,
                image_index=0,
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            )

            state = list(next_state)
            episode_rewards.append(reward)
            step += 1

            if SHOW_IMAGE:
                env.show_image(state)

        if len(agent.memory) > 0:
            agent.update(show=True)

        final_metrics = env.get_metrics(state)
        mean_reward = float(np.mean(episode_rewards)) if episode_rewards else final_metrics["reward"]

        reward_record.append([mean_reward])
        done_record.append(final_metrics["is_success"])
        cate_record.append(final_metrics["cate_accuracy"])
        hori_record.append(final_metrics["hori_accuracy"])
        vert_record.append(final_metrics["vert_accuracy"])

        if env.epsilon > EPSILON_MIN:
            env.epsilon = max(EPSILON_MIN, env.epsilon * env.epsilon_gamma)

        torch.save(agent.model.state_dict(), ACTOR_MODEL_PATH)
        torch.save(agent.critic_model.state_dict(), CRITIC_MODEL_PATH)
        save_log("reward" + FILE_NAME, reward_record)
        save_log("done" + FILE_NAME, done_record)
        save_log("cate" + FILE_NAME, cate_record)
        save_log("hori" + FILE_NAME, hori_record)
        save_log("vert" + FILE_NAME, vert_record)
        plot_reward_curve(reward_record, done_record, FILE_NAME)

        print(
            f"Episode {epoch + 1}/{epoch_num} | steps={step} | reward={mean_reward:.3f} | "
            f"success={int(final_metrics['is_success'])} | cate={final_metrics['cate_accuracy']:.3f} | "
            f"hori={final_metrics['hori_accuracy']:.3f} | vert={final_metrics['vert_accuracy']:.3f}"
        )


if __name__ == "__main__":
    set_seed(SEED)
    data = load_data()
    env = build_env(data["train_x"], data["train_y"], epsilon=EPSILON)
    switcher = build_switcher(env)
    train(env, switcher, epoch_num=TOTAL_EPISODES)
