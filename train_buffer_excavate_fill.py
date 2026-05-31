import os
import random
from typing import List

import numpy as np
import torch

from agent.digger_agent import DiggerAgent
from agent.filler_agent import FillerAgent
from env.buffer_excavate_fill_env import BufferExcavateFillEnv
from model_code.digger_model import DiggerModel
from model_code.filler_model import FillerModel
from utils.utils import plot_reward_curve, save_log


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_X_PATH = "dataset/train_img_48gap_33-001.npy"
TRAIN_Y_PATH = "dataset/train_label_48gap_33.npy"
MODEL_DIR = "model"
MODEL_BACKBONE = "modulator"
TOTAL_EPISODES = 100
SHOW_IMAGE = False
LOAD_MODEL = False
DIGGER_MODEL_PATH = os.path.join(MODEL_DIR, f"buffer_excavate_digger_{MODEL_BACKBONE}.pth")
FILLER_MODEL_PATH = os.path.join(MODEL_DIR, f"buffer_excavate_filler_{MODEL_BACKBONE}.pth")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data():
    return np.load(TRAIN_X_PATH), np.load(TRAIN_Y_PATH)


def build_env(train_x: np.ndarray, train_y: np.ndarray) -> BufferExcavateFillEnv:
    return BufferExcavateFillEnv(
        train_x=train_x,
        train_y=train_y,
        image_num=2,
        dig_per_board=2,
        fill_strategy="parallel",
        initial_swap_num=5,
        device=DEVICE,
    )


def build_agents(env: BufferExcavateFillEnv):
    digger_model = DiggerModel(model_name=MODEL_BACKBONE).to(DEVICE)
    filler_model = FillerModel(model_name=MODEL_BACKBONE).to(DEVICE)
    digger_agent = DiggerAgent(digger_model, device=DEVICE)
    filler_agent = FillerAgent(filler_model, device=DEVICE)
    if LOAD_MODEL:
        if os.path.exists(DIGGER_MODEL_PATH):
            digger_agent.model.load_state_dict(torch.load(DIGGER_MODEL_PATH, map_location=DEVICE))
        if os.path.exists(FILLER_MODEL_PATH):
            filler_agent.model.load_state_dict(torch.load(FILLER_MODEL_PATH, map_location=DEVICE))
    return digger_agent, filler_agent


def run_episode(env: BufferExcavateFillEnv, digger: DiggerAgent, filler: FillerAgent, epoch: int) -> float:
    env.reset()
    digger.clean_memory()
    filler.clean_memory()

    for board_id in range(env.image_num):
        for _ in range(env.dig_per_board):
            observation = env.get_digger_observation(board_id)
            action, _ = digger.choose_action(observation)
            next_obs, reward, done, info = env.step_digger(board_id, action)
            digger.recording_memory(observation, action, reward, done)
            if SHOW_IMAGE:
                env.show_image()

    total_fill_reward = 0.0
    while env.stage == "filler":
        candidates = env.get_filler_candidates()
        if not candidates:
            break
        candidate_index, _ = filler.choose_candidate(candidates)
        candidate = candidates[candidate_index]
        _, reward, done, info = env.step_filler(candidate_index)
        filler.recording_memory(candidate, reward)
        total_fill_reward += reward
        if done:
            break

    digger_loss = digger.update(train_epochs=1, show=False)
    filler_loss = filler.update(train_epochs=1, show=False)
    metrics = env.get_metrics()
    print(
        f"Episode {epoch + 1} | score={metrics['score']:.3f} | done={metrics['done_accuracy']:.3f} | "
        f"digger_reward={metrics['digger_reward']:.3f} | filler_reward={metrics['filler_reward']:.3f}"
    )
    return metrics["score"]


def main() -> None:
    set_seed(42)
    train_x, train_y = load_data()
    env = build_env(train_x, train_y)
    digger, filler = build_agents(env)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs("result", exist_ok=True)
    os.makedirs("json", exist_ok=True)

    reward_record: List[List[float]] = []
    done_record: List[List[float]] = []
    for epoch in range(TOTAL_EPISODES):
        score = run_episode(env, digger, filler, epoch)
        metrics = env.get_metrics()
        reward_record.append([score])
        done_record.append([metrics["done_accuracy"]])
        torch.save(digger.model.state_dict(), DIGGER_MODEL_PATH)
        torch.save(filler.model.state_dict(), FILLER_MODEL_PATH)
        save_log("buffer_excavate_fill_reward", reward_record)
        save_log("buffer_excavate_fill_done", done_record)
        plot_reward_curve(reward_record, done_record, "_buffer_excavate_fill")


if __name__ == "__main__":
    main()
