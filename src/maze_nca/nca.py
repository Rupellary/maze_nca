import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from maze_nca.config import EnvConfig

def make_perception_filters() -> tf.Tensor:
    """
    Generates convolution filters including sobels, identity, and laplacian to be applied depthwise.

    Parameters
    ----------
    None

    Returns
    ----------
    filters : tf.Tensor
        Tensor of shape (3, 3, 1, 4) with stack of filters
    """
    sobel_x = tf.constant([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    sobel_y = tf.constant([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])
    identity = tf.constant([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])
    laplacian = tf.constant([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1]
    ])
    filters = tf.stack([sobel_x, sobel_y, identity, laplacian], axis=-1) # shape: (3, 3, 4)
    return tf.expand_dims(filters, axis=2) # shape: (3, 3, 1, 4)


class NCAModel(tf.keras.Model):
    def __init__(
        self,
        living_channels: int,
        num_neurons: int,
        death_threshold: float,
        learned_filters: bool = False,
        num_learned_filters: int = 10,
    ):
        super().__init__()

        self.living_channels = living_channels
        self.num_neurons = num_neurons
        self.death_threshold = death_threshold
        self.learned_filters = learned_filters
        self.num_learned_filters = num_learned_filters

        # Convolution for perceiving neighbors, environment, and self
        if learned_filters: # Learn cross-channel filters
            self.perceive = layers.Conv2D(
                filters=self.num_learned_filters,
                kernel_size=3,
                padding="same",
                #use_bias=False
            )
            self.num_filters = self.num_learned_filters

        else: # Use depthwise sobel, identity, and laplacian filters
            perception_filters: tf.Tensor = make_perception_filters()
            H, W, _, F = perception_filters.shape

            C_perceptible: int = self.living_channels + 3 # (goal_distance, walls, goal)
            filters = tf.tile(perception_filters, (1, 1, C_perceptible, 1))  # shape: (3, 3, C_perceptible, F)

            self.perceive = layers.DepthwiseConv2D(
                kernel_size=(H, W),
                depth_multiplier=F,
                padding='same',
                use_bias=False,
                trainable=False,
                depthwise_initializer=filters.numpy() #tf.constant_initializer(filters)
            )
            self.num_filters = C_perceptible * F # F filters for each perceptible channel

        # MLP for interpreting and responding to perceptions
        self.react = keras.Sequential([
            keras.Input((self.num_filters,),), # input filter outputs from convolution
            layers.Dense(self.num_neurons, activation="relu"), # hidden layer
            layers.Dense(living_channels) # output delta state
        ])


    def egg(
        self,
        env : tf.Tensor,
        config : EnvConfig
    ) -> tf.Tensor:
        """
        Initilizes living channels and adds them to task tensor. Sets all living channels to a common constant only at the start.

        Parameters
        ----------
        self : NCAModel
            NCA
            Attributes used: [
                living_channels
            ]
        env : tf.Tensor
            Task environment tensor with all channels other than the living ones
        config : EnvConfig
            Config object specifying simulation parameters
            Attributes used: [
                idx_start, live_init
            ]

        Returns
        ----------
        world : tf.Tensor
            Tensor of shape (H, W, C) containing input channels + living channels
        """

        # Locate start
        start_mask = tf.cast(env[..., config.idx_start:config.idx_start+1], tf.float32) # shape: (H, W, 1)
        # Broadcast across living channels
        living_channels = start_mask * config.live_init # shape: (H, W, 1)
        living_channels = tf.tile(living_channels, [1, 1, self.living_channels]) # shape: (H, W, living_channels)
        # Combine with non-living environment
        return tf.concat([living_channels, env], axis=-1) # shape: (H, W, all_channels)


    # def embryogenesis


    def call(
        self,
        world : tf.Tensor,
        config : EnvConfig
    ) -> tf.Tensor:
        """
        Performs one step of the CA

        Parameters
        ----------
        self : NCAModel
            NCA
            Attributes used: [
                percieve, react,
                death_threshold
            ]
        world : tf.Tensor
            State tensor with all channels
        config : EnvConfig
            Config object specifying simulation parameters
            Attributes used: [
                sl_perceptible, sl_living, sl_non_living
                idx_obstacles, idx_alive,
                stochastic_update, update_rate
            ]

        Returns
        ----------
        new_state : tf.Tensor
            Updated state tensor with all channels
        """

        perceptible_world = world[..., config.sl_perceptible] # just perceptible channels
        alterable_world = world[..., config.sl_living] # just living channels
        task_env = world[..., config.sl_non_living] # just non-living channels

        # --- Perceive Neighbors (convolve into feature maps) ---
        perceived = self.perceive(perceptible_world) # shape: (B, H, W, num_filters)

        # --- Apply reaction at a per-cell level ---
        B, H, W, C = tf.unstack(tf.shape(perceived))
        # Flatten to channel vectors for each individual cell
        cells = tf.reshape(perceived, (-1, C)) # shape: (B * H * W, num_filters)
        # Apply DNN reaction on a cell-by-cell basis
        cell_reactions = self.react(cells) # shape: (B * H * W, num_living_channels)
        # Unflatten, restoring original shape
        reaction = tf.reshape(cell_reactions, (B, H, W, -1)) # shape: (B, H, W, num_living_channels)

        # --- Mask obstacle cells from being updated ---
        # Generate mask withs 0s where walls are and 1s elsewhere
        maze_mask = tf.cast(1 - world[..., config.idx_obstacles], reaction.dtype) # shape: (B, H, W, 1)
        # Mask wall locations from update
        reaction *= maze_mask # shape: (B, H, W, num_living_channels)

        # --- Stochastic updating ---
        if config.stochastic_update:
            # Generate random binary mask
            stoch_mask = tf.cast(
                tf.random.uniform((B, H, W, 1)) < config.update_rate,
                dtype=reaction.dtype
            ) # shape: (B, H, W, 1)
            # Stop random cells from updating
            reaction *= stoch_mask # shape: (B, H, W, num_living_channels)

        # --- Update living channel states ---
        new_living_state = alterable_world + reaction # shape: (B, H, W, living_channels)

        # --- Apply death threshold ---
        # Generate binary mask with 0s for cells below death threshold
        alive_mask = new_living_state[..., config.idx_alive] >= self.death_threshold # shape: (B, H, W)
        alive_mask = tf.cast(tf.expand_dims(alive_mask, -1), new_living_state.dtype) # shape: (B, H, W, 1)
        # Broadcast to set all living channels to 0 in death locations
        new_living_state *= alive_mask # shape: (B, H, W, living_channels)

        # --- Bound states between 0 and 1 in a way that is differentiable ---
        new_living_state = (tf.tanh(new_living_state) + 1.0) / 2.0 # shape: (B, H, W, living_channels)

        # --- Recombine living and non-living channels ---
        new_state = tf.concat([new_living_state, task_env], axis=3) # shape: (B,H,W,living+non_living)
        return new_state