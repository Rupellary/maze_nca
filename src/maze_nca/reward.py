import tensorflow as tf
from maze_nca.config import EnvConfig


@tf.function
def softmax_reward(
    world: tf.Tensor,
    config: EnvConfig
) -> tf.Tensor:
    """

    Parameters
    ----------

    Returns
    ----------
    """

    alive_channel = world[..., config.idx_alive] # shape: (B, H, W)
    # Count alive cells
    alive_mask = tf.cast(alive_channel > 0, tf.float32) # shape: (B, H, W)
    total_activity = tf.reduce_sum(alive_mask, axis=[1, 2]) # shape: (B)

    # Flatten solution proximity channel
    solution_proximity = world[..., config.idx_problem_distance] # shape: (B, H, W)
    B, H, W = tf.unstack(tf.shape(solution_proximity))
    flat = tf.reshape(solution_proximity, [B, H*W]) # shape: (B, H*W)
    # Apply softmax for smooth normalized reward with an easily tunable non-linearity
    softmax_flat = tf.nn.softmax(flat / config.reward_temperature, axis=-1) # shape: (B, H*W)
    softmaxed_reward = tf.reshape(softmax_flat, [B, H, W]) # shape: (B, H, W)
    # Upscale reward so that the gradients aren't too small. Multiplying by num_cells helps scalability
    num_cells = tf.cast(H * W, softmaxed_reward.dtype)
    softmax_scaled_reward = softmaxed_reward * num_cells

    if config.alive_scaled:
        claimed_reward = softmax_scaled_reward * alive_channel # shape: (B, H, W)
    else:
        claimed_reward = softmax_scaled_reward * alive_mask # shape: (B, H, W)

    task_rewards = tf.reduce_sum(claimed_reward, axis=[1, 2]) # shape: (B)
    task_rewards /= (total_activity * config.activity_cost + config.denom_epsilon) # shape: (B)
    avg_task_reward = tf.reduce_mean(task_rewards, axis=[0]) # shape: (scaler)
    return avg_task_reward