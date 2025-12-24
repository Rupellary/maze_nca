from dataclasses import dataclass
from typing import Optional, Callable, Any, Dict
import tensorflow as tf

@dataclass(frozen=True)
class EnvConfig:

    # Channel index aliases: 
    # These are *descriptive* references, changing them will break the code
    idx_alive: int = 0
    sl_living: slice = slice(0, -5)
    idx_goal_distance: int = -5
    idx_obstacles: int = -4
    idx_goal: int = -3
    idx_start: int = -2
    idx_problem_distance: int = -1

    sl_perceptible: slice = slice(0, -2)
    sl_non_living: slice = slice(-5, None)

    # Maze Size
    max_height: int = 15
    max_width: int = 15

    # Single Wall Params
    hole_size: int = 3

    # Value Iteration
    VI_goal_reward: float = 1
    VI_step_cost: float = 1
    VI_gamma: float = 0.9
    VI_theta: float = 1e-4
    VI_max_iters: int = int(1e3)

    # NCA Params
    death_threshold: float = 0.01,
    num_living_channels: int = 8,

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

    def get_task_shape(self) -> tuple[int]:
        """
        Returns tuple with H, W, C of task tensors
        """
        return (self.max_height, self.max_width, self.num_living_channels+5)

    def to_tf_task_cfg(self) -> Dict[str, Any]:
        """
        Converts task generator configs to graph-friendly dict
        """
        tf_task_cfg: Dict[str, Any] = {}

        # Channel index aliases
        tf_task_cfg["idx_alive"] = tf.constant(self.idx_alive, tf.int32)
        tf_task_cfg["idx_goal_distance"] = tf.constant(self.idx_goal_distance, tf.int32)
        tf_task_cfg["idx_obstacles"] = tf.constant(self.idx_obstacles, tf.int32)
        tf_task_cfg["idx_goal"] = tf.constant(self.idx_goal, tf.int32)
        tf_task_cfg["idx_start"] = tf.constant(self.idx_start, tf.int32)
        tf_task_cfg["idx_problem_distance"] = tf.constant(self.idx_problem_distance, tf.int32)

        tf_task_cfg["sl_living"] = self.sl_living
        tf_task_cfg["sl_perceptible"] = self.sl_perceptible
        tf_task_cfg["sl_non_living"] = self.sl_non_living

        # Maze size
        tf_task_cfg["max_height"] = tf.constant(self.max_height, tf.int32)
        tf_task_cfg["max_width"] = tf.constant(self.max_width, tf.int32)

        # Single wall params
        tf_task_cfg["hole_size"] = tf.constant(self.hole_size, tf.int32)

        # Value Iteration parameters
        tf_task_cfg["value_iteration_cfg"] = {
            "goal_reward": tf.constant(self.VI_goal_reward, tf.float32),
            "step_cost": tf.constant(self.VI_step_cost, tf.float32),
            "gamma": tf.constant(self.VI_gamma, tf.float32),
            "theta": tf.constant(self.VI_theta, tf.float32),
            "max_iters": tf.constant(self.VI_max_iters, tf.int32),
        }

        return tf_task_cfg