import copy
import os

import numpy as np
import torch

from agent.hierachy_buffer_switcher import Buffer_switcher
from agent.hierachy_decider import Decider
from agent.hierachy_local_switcher import Local_switcher
from env.hierachy_env import HierachyEnv, env
from model_code.hierachy_models import Buffer_switcher_model, Decider_model, Local_switcher_model
from utils.hierachy_config import (
    AGENT_EPOCHS,
    BATCH_SIZE,
    DEVICE,
    ENTROPY_WEIGHT,
    EPOCH_NUM,
    EPSILON,
    EPSILON_GAMMA,
    GAMMA,
    LOAD_MODEL,
    MAX_STEP,
    MODEL_NAME,
    SHOW_IMAGE,
    SWAP_NUM,
    TRAIN_PER_STEP,
    TRAIN_X_PATH,
    TRAIN_Y_PATH,
)


def load_data():
    train_x = np.load(TRAIN_X_PATH)
    train_y = np.load(TRAIN_Y_PATH)
    print(f"Data shape: x {train_x.shape}, y {train_y.shape}")
    return train_x, train_y


def update(decider, local_switcher, buffer_switcher, show=False):
    decider.update(show)
    local_switcher.update(show)
    buffer_switcher.update(show)


def clean_memory(decider, local_switcher, buffer_switcher):
    decider.clean_memory()
    local_switcher.clean_memory()
    buffer_switcher.clean_memory()


def save_models(decider, local_switcher, buffer_switcher):
    torch.save(decider.actor_model.state_dict(), os.path.join("model", "Decider_actor" + MODEL_NAME))
    torch.save(decider.critic_model.state_dict(), os.path.join("model", "Decider_critic" + MODEL_NAME))
    torch.save(buffer_switcher.model.state_dict(), os.path.join("model", "Buffer_switcher" + MODEL_NAME))
    torch.save(local_switcher.model.state_dict(), os.path.join("model", "Local_switcher" + MODEL_NAME))


def load_models(decider, local_switcher, buffer_switcher):
    decider.actor_model.load_state_dict(torch.load(os.path.join("model", "Decider_actor" + MODEL_NAME)))
    decider.critic_model.load_state_dict(torch.load(os.path.join("model", "Decider_critic" + MODEL_NAME)))
    buffer_switcher.model.load_state_dict(torch.load(os.path.join("model", "Buffer_switcher" + MODEL_NAME)))
    buffer_switcher.main_model = copy.deepcopy(buffer_switcher.model)
    local_switcher.model.load_state_dict(torch.load(os.path.join("model", "Local_switcher" + MODEL_NAME)))
    local_switcher.main_model = copy.deepcopy(local_switcher.model)


def build_components(load_train_data_flag=True):
    train_x, train_y = load_data() if load_train_data_flag else (None, None)
    environment = HierachyEnv(
        train_x=train_x,
        train_y=train_y,
        gamma=GAMMA,
        image_num=2,
        buffer_size=1,
        epsilon=EPSILON,
        epsilon_gamma=EPSILON_GAMMA,
    )
    decider_actor = Decider_model(
        fen_model_hidden1=512,
        fen_model_hidden2=512,
        outsider_hidden=512,
        hidden_1=512,
        hidden_2=512,
        action_num=2,
    ).to(DEVICE)
    decider_critic = Decider_model(
        fen_model_hidden1=512,
        fen_model_hidden2=512,
        outsider_hidden=512,
        hidden_1=1024,
        hidden_2=512,
        action_num=1,
    ).to(DEVICE)
    decider = Decider(
        memory_size=2000,
        actor=decider_actor,
        critic=decider_critic,
        env=environment,
        action_num=2,
        batch_size=BATCH_SIZE,
        entropy_weight=ENTROPY_WEIGHT,
        train_epoch=AGENT_EPOCHS,
    )
    buffer_switcher_model = Buffer_switcher_model(
        hidden_size1=512,
        hidden_size2=512,
        outsider_hidden_size=512,
        action_num=8,
    ).to(DEVICE)
    buffer_switcher = Buffer_switcher(
        memory_size=2000,
        model=buffer_switcher_model,
        action_num=8,
        batch_size=BATCH_SIZE,
        train_epoch=AGENT_EPOCHS,
        env=environment,
    )
    local_switcher_model = Local_switcher_model(
        fen_model_hidden1=512,
        fen_model_hidden2=512,
        hidden1=512,
        hidden2=512,
        action_num=1,
    ).to(DEVICE)
    local_switcher = Local_switcher(
        memory_size=2000,
        gamma=environment.gamma,
        batch_size=BATCH_SIZE,
        action_num=28,
        env=environment,
        model=local_switcher_model,
    )
    return environment, decider, buffer_switcher, local_switcher


def run_maze(env: env, decider: Decider, buffer_switcher: Buffer_switcher, local_switcher: Local_switcher, load_flag=True, epoch_num=500):
    if load_flag:
        load_models(decider=decider, local_switcher=local_switcher, buffer_switcher=buffer_switcher)

    for i in range(epoch_num):
        if i > 300:
            max_step, swap_num = MAX_STEP[3], SWAP_NUM[3]
        elif i > 200:
            max_step, swap_num = MAX_STEP[2], SWAP_NUM[2]
        elif i > 100:
            max_step, swap_num = MAX_STEP[1], SWAP_NUM[1]
        else:
            max_step, swap_num = MAX_STEP[0], SWAP_NUM[0]

        initial_perm = env.summon_permutation_list(swap_num)
        permutation_list = [initial_perm[j * (env.piece_num - 1) : (j + 1) * (env.piece_num - 1)] for j in range(env.image_num)]
        buffer = [-1] * env.buffer_size
        done_list = [False] * env.image_num
        reward_sum_list = [[] for _ in range(env.image_num)]
        termination_list = [0 for _ in range(env.image_num)]
        step = 0
        done = False
        clean_memory(decider, local_switcher, buffer_switcher)
        pending_transitions_decider = {j: None for j in range(env.image_num)}
        pending_transitions_local_switcher = {j: None for j in range(env.image_num)}
        pending_transitions_buffer_switcher = {j: None for j in range(env.image_num)}
        decider.trace_start_point = decider.memory_counter
        perm_with_buf = None

        while not done and step < max_step:
            do_list = []
            model_action = [0 for _ in range(env.image_num)]

            for j in range(env.image_num):
                if done_list[j]:
                    continue
                if termination_list[j] >= 40:
                    termination_list[j] = 0
                perm_with_buf = permutation_list[j] + buffer
                do_list.append(j)

                if pending_transitions_decider[j] is not None:
                    prev_state, prev_other_state, prev_action, prev_log_prob, prev_reward = pending_transitions_decider[j]
                    decider.recording_memory(
                        image_id=env.image_id,
                        image_index=j,
                        other_image_index=(j + 1) % env.image_num,
                        state=prev_state,
                        other_state=prev_other_state,
                        action=prev_action,
                        log_prob=prev_log_prob,
                        reward=prev_reward,
                        next_state=perm_with_buf,
                        next_other_state=permutation_list[(j + 1) % env.image_num],
                        done=done_list[j],
                    )
                    pending_transitions_decider[j] = None

                if pending_transitions_local_switcher[j] is not None:
                    prev_state, prev_action, prev_next_state, prev_reward, prev_done = pending_transitions_local_switcher[j]
                    local_switcher.recording_memory(
                        image_id=env.image_id,
                        image_index=j,
                        state=prev_state,
                        action=prev_action,
                        reward=prev_reward,
                        next_state=prev_next_state,
                        done=prev_done,
                    )
                    pending_transitions_local_switcher[j] = None

                if pending_transitions_buffer_switcher[j] is not None:
                    state, action, reward, transition_done = pending_transitions_buffer_switcher[j]
                    buffer_switcher.recording_memory(
                        image_id=env.image_id,
                        image_index=j,
                        state=state,
                        action=action,
                        reward=reward,
                        next_state=perm_with_buf,
                        done=transition_done,
                    )
                    pending_transitions_buffer_switcher[j] = None

                image, outsider = env.get_image(perm_with_buf, image_index=j)
                mask = env.get_mask(perm_with_buf)
                decider_action, decider_log_prob = decider.act(current_image=image, outsider_piece=outsider, mask=mask)
                pending_transitions_decider[j] = (
                    permutation_list[j],
                    permutation_list[(j + 1) % env.image_num],
                    decider_action,
                    decider_log_prob,
                )
                model_action[j] = decider_action

                if decider_action:
                    perm_with_buf_, action = buffer_switcher.act(image, outsider, perm_with_buf)
                    permutation_list[j] = copy.deepcopy(perm_with_buf[: len(perm_with_buf_) - env.buffer_size])
                    buffer = copy.deepcopy(perm_with_buf_[len(perm_with_buf_) - env.buffer_size :])
                    pending_transitions_buffer_switcher[j] = (perm_with_buf, action)
                else:
                    permutation_, action = local_switcher.act(permutation=permutation_list[j], image_index=j)
                    pending_transitions_local_switcher[j] = (permutation_list[j], action, permutation_)
                    permutation_list[j] = copy.deepcopy(permutation_)

            local_reward_list, consistency_reward_list, done_list = env.get_reward(permutation_list)

            if SHOW_IMAGE:
                env.show_image(permutation_list)

            for j in do_list:
                reward_sum_list[j].append(local_reward_list[j] + consistency_reward_list[j])
                prev_state, prev_other_state, prev_action, prev_log_prob = pending_transitions_decider[j]
                pending_transitions_decider[j] = (
                    prev_state,
                    prev_other_state,
                    prev_action,
                    prev_log_prob,
                    local_reward_list[j] + consistency_reward_list[j],
                )

                if model_action[j]:
                    state, action = pending_transitions_buffer_switcher[j]
                    pending_transitions_buffer_switcher[j] = (state, action, consistency_reward_list[j], done_list[j])
                else:
                    state, action, next_state = pending_transitions_local_switcher[j]
                    pending_transitions_local_switcher[j] = (state, action, next_state, local_reward_list[j], done_list[j])

            done = all(done_list)
            step += 1

            if step % TRAIN_PER_STEP == 0:
                update(decider=decider, local_switcher=local_switcher, buffer_switcher=buffer_switcher, show=False)
                env.load_image(image_num=env.image_num, id=env.image_id)
                decider.trace_start_point = decider.memory_counter

        for j in range(env.image_num):
            if pending_transitions_decider[j] is not None:
                prev_state, prev_other_state, prev_action, prev_log_prob, prev_reward = pending_transitions_decider[j]
                decider.recording_memory(
                    image_id=env.image_id,
                    image_index=j,
                    other_image_index=(j + 1) % env.image_num,
                    state=prev_state,
                    other_state=prev_other_state,
                    action=prev_action,
                    log_prob=prev_log_prob,
                    reward=prev_reward,
                    next_state=perm_with_buf,
                    next_other_state=permutation_list[(j + 1) % env.image_num],
                    done=done_list[j],
                )

            if pending_transitions_local_switcher[j] is not None:
                prev_state, prev_action, prev_next_state, prev_reward, prev_done = pending_transitions_local_switcher[j]
                local_switcher.recording_memory(
                    image_id=env.image_id,
                    image_index=j,
                    state=prev_state,
                    action=prev_action,
                    reward=prev_reward,
                    next_state=prev_next_state,
                    done=prev_done,
                )

            if pending_transitions_buffer_switcher[j] is not None:
                state, action, reward, transition_done = pending_transitions_buffer_switcher[j]
                buffer_switcher.recording_memory(
                    image_id=env.image_id,
                    image_index=j,
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=perm_with_buf,
                    done=transition_done,
                )

        print(
            f"Epoch: {i}, step: {step}, reward: "
            f"{[sum(reward_sum_list[j]) / len(reward_sum_list[j]) for j in range(len(reward_sum_list)) if len(reward_sum_list[j]) != 0]}"
        )
        print(f"Permutation list: {permutation_list}")
        if env.epsilon > 0.1:
            env.epsilon *= env.epsilon_gamma
        update(decider=decider, local_switcher=local_switcher, buffer_switcher=buffer_switcher, show=True)
        save_models(decider=decider, local_switcher=local_switcher, buffer_switcher=buffer_switcher)


def test_maze(env: env, decider: Decider, buffer_switcher: Buffer_switcher, local_switcher: Local_switcher, load_flag=True, swap_num=SWAP_NUM[0], max_step=MAX_STEP[0]):
    if load_flag:
        load_models(decider=decider, local_switcher=local_switcher, buffer_switcher=buffer_switcher)

    initial_perm = env.summon_permutation_list(swap_num)
    permutation_list = [initial_perm[j * (env.piece_num - 1) : (j + 1) * (env.piece_num - 1)] for j in range(env.image_num)]
    buffer = [-1] * env.buffer_size
    done_list = [False] * env.image_num
    step = 0

    while not all(done_list) and step < max_step:
        for j in range(env.image_num):
            if done_list[j]:
                continue
            perm_with_buf = permutation_list[j] + buffer
            image, outsider = env.get_image(perm_with_buf, image_index=j)
            mask = env.get_mask(perm_with_buf)
            decider_action, _ = decider.act(current_image=image, outsider_piece=outsider, mask=mask)
            if decider_action:
                perm_with_buf_, _ = buffer_switcher.act(image, outsider, perm_with_buf)
                permutation_list[j] = copy.deepcopy(perm_with_buf[: len(perm_with_buf_) - env.buffer_size])
                buffer = copy.deepcopy(perm_with_buf_[len(perm_with_buf_) - env.buffer_size :])
            else:
                permutation_, _ = local_switcher.act(permutation=permutation_list[j], image_index=j)
                permutation_list[j] = copy.deepcopy(permutation_)
        _, _, done_list = env.get_reward(permutation_list)
        if SHOW_IMAGE:
            env.show_image(permutation_list)
        step += 1

    print(f"Test step: {step}")
    print(f"Test permutation list: {permutation_list}")
    return permutation_list, done_list


def main():
    environment, decider, buffer_switcher, local_switcher = build_components()
    print(f"Device: {DEVICE}")
    run_maze(
        env=environment,
        decider=decider,
        local_switcher=local_switcher,
        buffer_switcher=buffer_switcher,
        epoch_num=EPOCH_NUM,
        load_flag=LOAD_MODEL,
    )


def main_test():
    environment, decider, buffer_switcher, local_switcher = build_components()
    print(f"Device: {DEVICE}")
    test_maze(
        env=environment,
        decider=decider,
        local_switcher=local_switcher,
        buffer_switcher=buffer_switcher,
        load_flag=True,
    )


if __name__ == "__main__":
    main()
