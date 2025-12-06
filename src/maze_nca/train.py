import tensorflow as tf
from typing import Optional, Callable
from maze_nca.nca import NCAModel
from maze_nca.config import EnvConfig
from tensorflow.keras.optimizers import AdamW

@tf.function
def train_step(
    nca: NCAModel,
    task: tf.Tensor,
    optimizer,
    config: EnvConfig
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
        optimizer for backprop
    config : EnvConfig
        Config object specifying simulation parameters
        Attributes used: [
            reward_function,
            rollout_steps,
            loss_gamma
        ]

    Returns:
    ----------
    total_loss : tf.Tensor
        Total loss from the rollout
    """

    compute_reward: Callable[[tf.Tensor], tf.Tensor] = config.reward_function

    x: tf.Tensor = nca.egg(task)
    r: tf.Tensor = compute_reward(x)
    T: int = config.rollout_steps
    discount: tf.constant = tf.constant(1.0)
    total_loss: tf.constant = tf.constant(0.0)

    with tf.GradientTape() as tape:

        # x = nca.embryogenesis(x)

        for _ in range(T):
            # Update state
            x = nca(x)
            # Compute state reward
            rt = compute_reward(x)
            # Compute negative delta reward (value will be minimized)
            neg_delta_r = r - rt
            r = rt
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
    config: EnvConfig
) -> list[float]:
    """
    Performs one rollout of NCA and applys backpropogation.

    Parameters:
    ----------
    nca : NCAModel
        Model to train
    config : EnvConfig
        Config object specifying simulation parameters
        Attributes used: [
            lr, weight_decay,
            epochs
        ]

    Returns:
    ----------
    history : list
        List containing the loss after each epoch
    """

    optimizer = AdamW(
        learning_rate=config.lr,
        weight_decay=config.weight_decay
    )

    history = []

    for i in range(config.epochs):

        loss = train_step(nca, task, optimizer, config)
        history.append(loss)

        if (i + 1) % 10 == 0:
            print(loss)

    return history