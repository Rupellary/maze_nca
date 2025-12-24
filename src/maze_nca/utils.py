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


def _mask_walls(
    env: tf.Tensor,
    config: EnvConfig
) -> tf.Tensor:
    """
    Helper function for deactivating living channels where there are walls
    """

    living_env = env[..., config.sl_living]
    wall_mask = 1 - env[..., config.idx_obstacles] # shape: (B, H, W)
    wall_mask = tf.expand_dims(wall_mask, axis=-1) # shape: (B, H, W, 1)
    living_env *= wall_mask
    return tf.concat([living_env, env[..., config.sl_non_living]], axis=-1)




def _fill_space(
    env: tf.Tensor,
    living_channels,
    config: EnvConfig
) -> tf.Tensor:
    """
    Creates state where living channels are active everywhere. Helper function for probing reward function.
    """

    B, H, W, C = tf.unstack(tf.shape(env))
    alive = tf.ones((B, H, W, living_channels), dtype=tf.float32)
    env = tf.concat([alive, env], axis=-1)
    env = _mask_walls(env, config)
    return env



def _half_fill_space(
    env: tf.Tensor,
    living_channels,
    config: EnvConfig
) -> tf.Tensor:
    """
    Creates state where living channels are half-active everywhere. Helper function for probing reward function.
    """

    B, H, W, C = tf.unstack(tf.shape(env))
    alive = tf.ones((B, H, W, living_channels), dtype=tf.float32)
    alive /= 2
    env = tf.concat([alive, env], axis=-1)
    env = _mask_walls(env, config)
    return env



def _empty_space(
    env: tf.Tensor,
    living_channels
) -> tf.Tensor:
    """
    Creates state where living channels are inactive everywhere. Helper function for probing reward function.
    """

    B, H, W, C = tf.unstack(tf.shape(env))
    alive = tf.zeros((B, H, W, living_channels), dtype=tf.float32)
    return tf.concat([alive, env], axis=-1)



def _goal_only(
    env : tf.Tensor,
    living_channels,
    config: EnvConfig
) -> tf.Tensor:
    """
    Creates state where living channels are active only at the goal location. Helper function for probing reward function.
    """

    # Locate goal
    goal_mask = tf.cast(env[..., config.idx_goal:config.idx_goal+1], tf.float32) # shape: (B, H, W, 1)
    # Broadcast across living channels
    alive = tf.tile(goal_mask, [1, 1, 1, living_channels]) # shape: (B, H, W, living_channels)
    # Combine with non-living environment
    return tf.concat([alive, env], axis=-1) # shape: (B, H, W, all_channels)




def benchmark_reward(
    config: EnvConfig,
    ca: NCAModel,
    batch_size: int,
    base_seed: int
) -> None:

    env = generate_batch(
        config.to_tf_task_cfg(),
        batch_size=batch_size,
        task_shape=config.get_task_shape(),
        base_seed=base_seed
    )

    start_batch = ca.egg(env, config)

    full_batch = _fill_space(env, config.num_living_channels, config)

    half_full_batch = _half_fill_space(env, config.num_living_channels, config)

    empty_batch = _empty_space(env, config.num_living_channels)

    goal_batch = _goal_only(env, config.num_living_channels, config)

    print(f"""
    Starting Reward: {softmax_reward(start_batch, config)}
    Goal Reward: {softmax_reward(goal_batch, config)}
    Space-Filled Reward: {softmax_reward(full_batch, config)}
    Half-Space-Filled Reward: {softmax_reward(half_full_batch, config)}
    Self-Destruct Reward: {softmax_reward(empty_batch, config)}
    """)