from agent.hierachy_buffer_switcher import Buffer_switcher
from agent.hierachy_decider import Decider
from agent.hierachy_local_switcher import Local_switcher
from env.hierachy_env import HierachyEnv, env
from model_code.hierachy_models import Buffer_switcher_model, Decider_model, Local_switcher_model, fen_model
from run_maze_hierachy import (
    build_components,
    clean_memory,
    load_data,
    load_models as load,
    main,
    main_test,
    run_maze,
    save_models as save,
    test_maze,
    update,
)
from utils.hierachy_config import *  # noqa: F401,F403
from utils.hierachy_utils import has_any_gradient, selective_load_state_dict


if __name__ == "__main__":
    main()
