import tensorflow as tf
from typing import Optional, Callable
from maze_nca.nca import NCAModel
from maze_nca.config import EnvConfig
from maze_nca.reward import softmax_reward
from tensorflow.keras.optimizers import AdamW

@tf.function
def train_step(
    nca: NCAModel,
    task: tf.Tensor,
    optimizer,
    config: EnvConfig,
    seed: tf.constant,
) -> tf.Tensor:
    """
    Performs one rollout of NCA and applies backpropogation.

    Parameters:
    ----------
    nca : NCAModel
        Model to train
    task : tf.Tensor
        Tensor with batch of mazes' non-living channels, shape (B, H, W, non_living)
    optimizer : AdamW
        Optimizer for backprop
    config : EnvConfig
        Config object specifying simulation parameters
        Attributes used: [
            reward_function,
            rollout_steps,
            loss_gamma
        ]
    seed : tf.constant <--- FINISH THIS

    Returns:
    ----------
    total_loss : tf.Tensor
        Total cumulative loss from the rollout
    """

    compute_reward = softmax_reward # <--- TEMPORARY REWARD FUNCTION ASSIGNMENT

    # Add living channels to task tensor
    world: tf.Tensor = nca.egg(task)

    # Intiliaze reward, discount, and cumulative loss
    previous_reward: tf.Tensor = compute_reward(world, config)
    discount: tf.constant = tf.constant(1.0) # discount will strengthen over rollout
    total_loss: tf.constant = tf.constant(0.0)

    with tf.GradientTape() as tape:

        # world = nca.embryogenesis(world)

        for _ in range(config.rollout_steps):
            # Update state
            world = nca(world, seed)
            # Compute state reward
            current_reward = compute_reward(world, config)
            # Compute negative delta reward (value will be minimized)
            neg_delta_r = previous_reward - current_reward
            previous_reward = current_reward
            # Apply discount
            loss = neg_delta_r * discount
            discount *= config.loss_gamma
            # Add step loss to total
            total_loss += loss

    grads = tape.gradient(total_loss, nca.trainable_variables)
    optimizer.apply_gradients(zip(grads, nca.trainable_variables))
    return total_loss


def train_loop(
    nca: NCAModel,
    task: tf.Tensor,
    optimizer,
    config: EnvConfig,
    seed: tf.constant,
) -> list[float]:
    """
    Performs one rollout of NCA and applys backpropogation.

    Parameters:
    ----------
    nca : NCAModel
        Model to train
    task : tf.Tensor                    <--- FINISH THIS
    optimizer :                         <--- FINISH THIS
    config : EnvConfig
        Config object specifying simulation parameters
        Attributes used: [
            lr, weight_decay,
            epochs
        ]
    seed : tf.constant                  <--- FINISH THIS

    Returns:
    ----------
    history : list
        List containing the loss after each epoch
    """

    history: list[float] = []

    for i in range(config.epochs):

        loss: float = train_step(nca, task, optimizer, config, seed)
        history.append(loss)

        # Print loss every 10 epochs
        if (i + 1) % 10 == 0:
            print(loss)

    return history