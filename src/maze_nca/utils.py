import matplotlib.pyplot as plt
import tensorflow as tf
from maze_nca.config import EnvConfig
from maze_nca.nca import NCAModel
from maze_nca.reward import softmax_reward
from maze_nca.task_generation import *

def plot_channel(tensor, channel, cmap="gray"):
    """
    Plot a single channel from an (H, W, C) tensor in grayscale.

    Parameters
    ----------
    tensor : tf.Tensor or np.ndarray
        Shape (H, W, C)
    channel : int
        Channel index to visualize
    cmap : str
        Matplotlib colormap (default: "gray")
    """

    if isinstance(tensor, tf.Tensor):
        tensor = tensor.numpy()  # convert to numpy for plotting

    # Extract channel
    img = tensor[..., channel]

    # Dynamic rescaling to [0,1] for display
    vmin, vmax = img.min(), img.max()
    if vmin == vmax:  # avoid div/0 if constant
        scaled = img * 0
    else:
        scaled = (img - vmin) / (vmax - vmin)

    plt.imshow(scaled, cmap=cmap, interpolation="nearest")
    plt.title(f"Channel {channel} (scaled)")
    plt.colorbar()
    plt.show()




# Generating states to probe reward function behavior


def mask_walls(
    env: tf.Tensor,
    config: EnvConfig
) -> tf.Tensor:
    """
    Helper function for deactivating living channels where there are walls
    """

    living_env = env[..., config.sl_living]
    wall_mask = 1 - env[..., config.idx_obstacles] # shape: (H, W)
    wall_mask = tf.expand_dims(wall_mask, axis=-1) # shape: (H, W, 1)
    living_env *= wall_mask
    return tf.concat([living_env, env[..., config.sl_non_living]], axis=-1)




def fill_space(
    env: tf.Tensor,
    living_channels,
    config: EnvConfig
) -> tf.Tensor:
    """
    Creates state where living channels are active everywhere. Helper function for probing reward function.
    """

    H, W, C = tf.unstack(tf.shape(env))
    alive = tf.ones((H, W, living_channels), dtype=tf.float32)
    env = tf.concat([alive, env], axis=-1)
    env = mask_walls(env, config)
    return env



def half_fill_space(
    env: tf.Tensor,
    living_channels,
    config: EnvConfig
) -> tf.Tensor:
    """
    Creates state where living channels are half-active everywhere. Helper function for probing reward function.
    """

    H, W, C = tf.unstack(tf.shape(env))
    alive = tf.ones((H, W, living_channels), dtype=tf.float32)
    alive /= 2
    env = tf.concat([alive, env], axis=-1)
    env = mask_walls(env, config)
    return env



def empty_space(
    env: tf.Tensor,
    living_channels
) -> tf.Tensor:
    """
    Creates state where living channels are inactive everywhere. Helper function for probing reward function.
    """

    H, W, C = tf.unstack(tf.shape(env))
    alive = tf.zeros((H, W, living_channels), dtype=tf.float32)
    return tf.concat([alive, env], axis=-1)



def goal_only(
    env : tf.Tensor,
    living_channels,
    config: EnvConfig
) -> tf.Tensor:
    """
    Creates state where living channels are active only at the goal location. Helper function for probing reward function.
    """

    # Locate goal
    goal_mask = tf.cast(env[..., config.idx_goal:config.idx_goal+1], tf.float32) # shape: (H, W, 1)
    # Broadcast across living channels
    alive = tf.tile(goal_mask, [1, 1, living_channels]) # shape: (H, W, living_channels)
    # Combine with non-living environment
    return tf.concat([alive, env], axis=-1) # shape: (H, W, all_channels)




def benchmark_reward(
    config: EnvConfig,
    ca: NCAModel,
    batch_size: int = 10,
    living_channels: int = 8,
) -> None:

    start_tasks = []
    full_tasks = []
    half_full_tasks = []
    empty_tasks = []
    goal_tasks = []

    for i in range(batch_size):
        env = generate_task(
            config.to_tf_task_cfg(),
            seed=i
        )

        start = ca.egg(env, config)
        start_tasks.append(start)

        full = fill_space(env, living_channels, config)
        full_tasks.append(full)

        half_full = half_fill_space(env, living_channels, config)
        half_full_tasks.append(half_full)

        empty = empty_space(env, living_channels)
        empty_tasks.append(empty)

        goal = goal_only(env, living_channels, config)
        goal_tasks.append(goal)

    start_batch = tf.stack(start_tasks, axis=0)
    full_batch = tf.stack(full_tasks, axis=0)
    half_full_batch = tf.stack(half_full_tasks, axis=0)
    empty_batch = tf.stack(empty_tasks, axis=0)
    goal_batch = tf.stack(goal_tasks, axis=0)

    print(f"""
    Starting Reward: {softmax_reward(start_batch, config)}
    Goal Reward: {softmax_reward(goal_batch, config)}
    Space-Filled Reward: {softmax_reward(full_batch, config)}
    Half-Space-Filled Reward: {softmax_reward(half_full_batch, config)}
    Self-Destruct Reward: {softmax_reward(empty_batch, config)}
    """)