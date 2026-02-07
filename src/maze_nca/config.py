from dataclasses import dataclass
from typing import Optional, Callable, Any, Dict
import tensorflow as tf

@dataclass(frozen=True)
class EnvConfig:

    # Channel index aliases: 
    # These are *descriptive* references, changing them will break the code
    idx_alive: int = 0
    idx_goal_distance: int = -5
    idx_obstacles: int = -4
    idx_goal: int = -3
    idx_start: int = -2
    idx_problem_distance: int = -1
    # Indices for easy slice creation (slices can't be serialized)
    idxs_living: tuple = (0, -5)
    idxs_perceptible: tuple = (0, -2)
    idxs_nonliving: tuple = (-5, None)

    # --- Space Specifications ---
    # Maze Size
    max_height: int = 15
    max_width: int = 15
    # Single Wall Task Params
    hole_size: int = 3

    # --- Value Iteration ---
    VI_goal_reward: float = 1
    VI_step_cost: float = 1
    VI_gamma: float = 0.9
    VI_theta: float = 1e-4
    VI_max_iters: int = int(1e3)

    # --- NCA ---
    num_neurons: int = 32
    num_living_channels: int = 8 
    death_threshold: float = 0.01 # for death masking
    update_rate: float = 0.5 # for stochastic updating
    avg_signal_threshold: float = 0.01 # for preventing spontaneous generation
    delta_limit: float = 0.5 # smoothens change over time
    signal_decay: float = 0.01 # adds bias towards inactivity in living channels
    # Embryogensis
    live_init: float = 0.5

    # --- Reward ---
    activity_cost: float = 1.0
    denom_epsilon: float = 1.0
    reward_temperature: float = 0.1

    # --- BPTT ---
    rollout_steps: int = 30
    epochs: int = 10
    loss_gamma: float = 0.9
    lr: float = 1e-3
    weight_decay: float = 1e-4


    # Playing around
    alive_scaled: bool = True


    def get_task_shape(self) -> tuple[int]:
        """
        Returns tuple with H, W, C of task tensors
        """

        C = self.num_living_channels + 5
        return self.max_height, self.max_width, C


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

        tf_task_cfg["idxs_living"] = self.idxs_living
        tf_task_cfg["idxs_perceptible"] = self.idxs_perceptible
        tf_task_cfg["idxs_nonliving"] = self.idxs_nonliving

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
    
    
    def to_tf_nca_cfg(self) -> Dict[str, Any]:
        """
        Converts nca configs to graph-friendly dict
        """

        tf_nca_cfg: Dict[str, Any] = {}

        # Architecture Specifications
        tf_nca_cfg['num_neurons'] = self.num_neurons
        tf_nca_cfg['num_living_channels'] = self.num_living_channels

        # Channel Indexes
        tf_nca_cfg['idx_alive'] = self.idx_alive
        tf_nca_cfg['idx_goal_distance'] = self.idx_goal_distance
        tf_nca_cfg['idx_obstacles'] = self.idx_obstacles
        tf_nca_cfg['idx_goal'] = self.idx_goal
        tf_nca_cfg['idx_start'] = self.idx_start
        tf_nca_cfg['idx_problem_distance'] = self.idx_problem_distance
        # Channel Slices
        tf_nca_cfg['idxs_living'] = self.idxs_living
        tf_nca_cfg['idxs_perceptible'] = self.idxs_perceptible
        tf_nca_cfg['idxs_nonliving'] = self.idxs_nonliving
        
        # NCA Params
        tf_nca_cfg['death_threshold'] = tf.constant(self.death_threshold, tf.float32)
        tf_nca_cfg['update_rate'] = tf.constant(self.update_rate, tf.float32)
        tf_nca_cfg['live_init'] = tf.constant(self.live_init, tf.float32)
        tf_nca_cfg['avg_signal_threshold'] = tf.constant(self.avg_signal_threshold, tf.float32)
        tf_nca_cfg['delta_limit'] = tf.constant(self.delta_limit, tf.float32)
        tf_nca_cfg['signal_decay'] = tf.constant(self.signal_decay, tf.float32)

        return tf_nca_cfg