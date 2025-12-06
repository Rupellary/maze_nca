from dataclasses import dataclass
from typing import Optional, Callable
import tensorflow as tf

@dataclass(frozen=True)
class EnvConfig:
    # Channel index aliases: cannot be shuffled because of how tasks are generated
    idx_alive: int = 0
    sl_living: slice = slice(0, -5)
    idx_goal_distance: int = -5
    idx_obstacles: int = -4
    idx_goal: int = -3
    idx_start: int = -2
    idx_problem_distance: int = -1

    sl_perceptible: slice = slice(0, -2)
    sl_non_living: slice = slice(-5, None)

    # NCA Params
    death_threshold: float = 0.01,
    num_living_channels: int = 8,

    # Value Iteration
    VI_goal_reward: float = 1
    VI_step_cost: float = 1
    VI_gamma: float = 0.9
    VI_theta: float = 1e-4
    VI_max_iters: int = int(1e3)

    # Embryogensis
    live_init: float = 0.5

    # Reward
    activity_cost: float = 1.0,
    denom_epsilon: float = 1.0,
    reward_temperature: float = 0.1

    # BPTT
    reward_function: Callable[[tf.Tensor], tf.Tensor] = None,#softmax_reward,
    rollout_steps: int = 30,
    loss_gamma: float = 0.9,
    lr: float = 1e-3,
    weight_decay: float = 1e-4


    # Playing around
    alive_scaled: bool = True