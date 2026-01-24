import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Dict, Any

def as_slice(slice_tuple):
    return slice(*slice_tuple)

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
        config: dict[str, Any]
    ):
        super().__init__()
        self.config = config

        # Use depthwise sobel, identity, and laplacian filters
        perception_filters: tf.Tensor = make_perception_filters()
        H, W, _, F = perception_filters.shape

        C_perceptible: int = self.config['num_living_channels'] + 3 # (goal_distance, walls, goal)
        filters = tf.tile(perception_filters, (1, 1, C_perceptible, 1))  # shape: (3, 3, C_perceptible, F)

        # --- Perception ---
        # Setting up convolutions for gathering neighborhood information
        self.perceive = layers.DepthwiseConv2D(
            kernel_size=(H, W),
            depth_multiplier=F,
            padding='same',
            use_bias=False,
            trainable=False,
            depthwise_initializer=filters.numpy() #tf.constant_initializer(filters)
        )
        num_filters = C_perceptible * F # F filters for each perceptible channel

        # --- Action ---
        # Setting up MLP for interpreting and responding to perceptions
        self.react = keras.Sequential([
            keras.Input((num_filters,),), # input filter outputs from convolutions
            layers.Dense(self.config['num_neurons'], activation="relu"), # hidden layer
            layers.Dense(self.config['num_living_channels']) # output delta state
        ])


    def egg(
        self,
        env: tf.Tensor,
    ) -> tf.Tensor:
        """
        Initilizes living channels and adds them to task tensor. Sets all living channels to a common constant only at the start.

        Parameters
        ----------
        self : NCAModel
            Neural Cellular Automata
            Config keys used: [
                idx_start,
                live_init,
                num_living_channels
            ]
        env : tf.Tensor
            Task environment tensor with all channels other than the living ones

        Returns
        ----------
        world : tf.Tensor
            Tensor of shape (H, W, C) containing input channels + living channels
        """

        # Locate start
        start_mask = tf.cast(env[..., self.config['idx_start']:self.config['idx_start']+1], tf.float32) # shape: (B, H, W, 1)
        # Broadcast across living channels
        living_channel = start_mask * self.config['live_init'] # shape: (B, H, W, 1)
        living_channels = tf.repeat(living_channel, repeats=self.config['num_living_channels'], axis=-1) # shape: (B, H, W, num_living_channels)
        # Combine with non-living environment
        return tf.concat([living_channels, env], axis=-1) # shape: (B, H, W, all_channels)


    # def embryogenesis


    def call(
        self,
        world: tf.Tensor,
        seed: tf.Tensor
    ) -> tf.Tensor:
        """
        Performs one step of the CA.

        Parameters
        ----------
        self : NCAModel
            Neural Cellular Automata
            Attributes used: [
                percieve, 
                react
            ]
            Config keys used: [
                idxs_perceptible, idxs_living, idxs_non_living
                idx_obstacles, idx_alive,
                update_rate, death_threshold
            ]
        world : tf.Tensor
            State tensor with all channels
        seed : tf.Tensor
            Seed for stochastic updating with stateless rng. Shape: (2)

        Returns
        ----------
        new_state : tf.Tensor
            Updated state tensor with all channels
        """

        sl_perceptible = slice(*self.config['idxs_perceptible'])
        sl_living = slice(*self.config['idxs_living'])
        sl_nonliving = slice(*self.config['idxs_nonliving'])

        perceptible_world = world[..., sl_perceptible] # just perceptible channels
        alterable_world = world[..., sl_living] # just living channels
        task_env = world[..., sl_nonliving] # just non-living channels


        # --- Perceive Neighbors (convolve into feature maps) ---
        perceived = self.perceive(perceptible_world) # shape: (B, H, W, num_filters)


        # --- Apply reaction at a per-cell level ---
        B, H, W, C = tf.unstack(tf.shape(perceived))
        # Flatten to channel vectors for each individual cell
        # Essentially a list of computed neighbor information for each cell
        cell_neighbor_info = tf.reshape(perceived, (-1, C)) # shape: (B * H * W, num_filters)
        # Apply DNN reaction on a cell-by-cell basis
        cell_reactions = self.react(cell_neighbor_info) # shape: (B * H * W, num_living_channels)
        # Unflatten, restoring original shape, distributing reactions spatially
        reaction = tf.reshape(cell_reactions, (B, H, W, -1)) # shape: (B, H, W, num_living_channels)


        # --- Mask obstacle cells from being updated ---
        # Generate mask withs 0s where walls are and 1s elsewhere
        wall_channel = slice(
            self.config['idx_obstacles'], 
            self.config['idx_obstacles']+1
        ) # must slice to preserve channel dim
        maze_mask = tf.cast(1 - world[..., wall_channel], reaction.dtype) # shape: (B, H, W, 1)
        # Mask wall locations from update
        reaction *= maze_mask # shape: (B, H, W, num_living_channels)


        # --- Stochastic updating ---
        # Generate random binary mask
        stoch_mask = tf.cast(
            tf.random.stateless_uniform((B, H, W, 1), seed) < self.config['update_rate'],
            dtype=reaction.dtype
        ) # shape: (B, H, W, 1)
        # Stop random cells from updating
        reaction *= stoch_mask # shape: (B, H, W, num_living_channels)

        # --- Prevent spontaneous generation, requiring signals propogate through space ---
        # -- Prevent spontaneous generation of life --
        # Can only become alive if there is life in the neighborhood
        neighborhood_kernel = tf.ones(
            (3, 3, 1, 1), 
            dtype=reaction.dtype
        )
        alive_channel = slice(
            self.config['idx_alive'], 
            self.config['idx_alive']+1
        ) # must slice to preserve channel dim
        # Sum aliveness channel in 3x3 neighborhood
        neighborhood_aliveness = tf.nn.conv2d(
            world[..., alive_channel],
            neighborhood_kernel,
            strides=1,
            padding='SAME'
        ) # shape: (B, H, W, 1)
        # Check for surrounding life
        life_adjacent_mask = neighborhood_aliveness >= self.config['death_threshold'] # shape: (B, H, W, 1)
        life_adjacent_mask = tf.cast(life_adjacent_mask, reaction.dtype)
        # Mask life in nonliving areas, preventing spontaneous generation
        life_reaction = reaction[..., alive_channel] * life_adjacent_mask # shape: (B, H, W, 1)

        # -- Prevent spontaneous generation of signals --
        # Other channels can synthesize one another but cannot arise from nothing
        neighborhood_kernel = tf.ones(
            (3, 3, self.config['num_living_channels'], 1), 
            dtype=reaction.dtype
        )
        neighborhood_presence = tf.nn.conv2d(
            world[..., sl_living],
            neighborhood_kernel,
            strides=1,
            padding='SAME'
        ) # shape: (B, H, W, 1)
        signal_threshold = self.config['avg_signal_threshold'] * self.config['num_living_channels']
        signal_adjacent_mask = neighborhood_presence >= signal_threshold # shape: (B, H, W, 1)
        signal_adjacent_mask = tf.cast(signal_adjacent_mask, reaction.dtype)
        signal_reaction = reaction[..., 1:] * signal_adjacent_mask # shape: (B, H, W, num_living_channels-1)

        # -- Recombine masked alive channel and masked signal channels --
        reaction = tf.concat([life_reaction, signal_reaction], axis=-1)


        # --- Dampen deltas ---
        reaction = self.config['delta_limit'] * tf.tanh(reaction / self.config['delta_limit']) # shape: (B, H, W, num_living_channels)


        # --- Apply signal decay ---
        alterable_world *= (1.0 - self.config['signal_decay']) # shape: (B, H, W, num_living_channels)


        # --- Update living channel states ---
        new_living_state = alterable_world + reaction # shape: (B, H, W, num_living_channels)


        # --- Apply death threshold ---
        # Generate binary mask with 0s for cells below death threshold
        alive_mask = new_living_state[..., alive_channel] >= self.config['death_threshold'] # shape: (B, H, W, 1)
        alive_mask = tf.cast(alive_mask, new_living_state.dtype)
        # Broadcast to set all living channels to 0 in death locations
        new_living_state *= alive_mask # shape: (B, H, W, num_living_channels)


        # --- Prevent negative signals ---
        new_living_state = tf.maximum(new_living_state, 0)


        # --- Recombine living and non-living channels ---
        new_state = tf.concat([new_living_state, task_env], axis=3) # shape: (B, H, W, living+non_living)
        return new_state