import tensorflow as tf
from maze_nca.config import EnvConfig
from typing import Optional, Callable

@tf.function
def make_bordered_space(
    height: int,
    width: int
) -> tf.Tensor:
    """
    Generates tensor with shape (height, width) with 1s on the borders and 0s on the inside.

    Parameters
    ----------
    height : int
        Height of the space
    width: int
        Width of the space

    Returns
    ----------
    space : tf.Tensor
        Tensor of shape (height, width) with 1s on the borders and 0s on the inside.
    """

    # Generate space without border
    inner = tf.zeros((height - 2, width - 2), dtype=tf.int32) # shape: (H-2, W-2)
    # Pad with border
    space = tf.pad(inner, paddings=[[1, 1], [1, 1]], constant_values=1) # shape: (H, W)
    return space



@tf.function
def make_single_wall_env(
    height: int,
    width: int,
    hole_size: Optional[int] = 1,
    seed: Optional[int] = None
) -> tf.Tensor:
    """
    Randomly generates an environment with a start and goal location and a single wall between them with a hole in the wall.

    Parameters
    ----------
    height : int
        height of the space
    width : int
        width of the space
    hole_size : int
        width of the hole in the wall
    seed : int
        random seed

    Returns
    ----------
    env : tf.Tensor
        Tensor with shape (H, W, 3) containing obstacle, start, and goal channels
    """

     # Establish random seed
    if seed is not None:
        tf.random.set_seed(seed)

    # Wall is always generated veritcally, randomly transpose to get horizontal walls
    # Randomly determine whether to transpose
    do_transpose = tf.random.uniform((), 0, 2, dtype=tf.int32)
    # If transposing at the end, use swapped dimensions
    H, W = tf.cond(
        do_transpose > 0,
        lambda: (width, height),
        lambda: (height, width)
    )

    # --- Generate obstacles channel with random vertical wall with hole ---
    # Initialize obstacles channel with border
    obstacles_channel: tf.Tensor = make_bordered_space(H, W)

    # -- Randomly place vertical wall -- (annoying in tf b/c immutable)
    # Define repeated variable
    wall_length: int = H-2
    # Find random location with at least 1 empty space between border and wall
    wall_col = tf.random.uniform((), 2, W-2, dtype=tf.int32)
    # Generate y coordinates for wall (all coordinates but borders)
    wall_y = tf.range(1, H - 1) # shape: (H-2,)
    # Generate x coordinates for wall (a bunch of the column coordinate)
    wall_x = tf.fill([wall_length], wall_col) # shape: (H-2,)
    # Combine into x, y pairs
    wall_indices = tf.stack([wall_y, wall_x], axis=1) # shape: (H-2, H-2)
    # Generate wall tensor (just a bunch of 1s)
    wall = tf.ones((wall_length,), dtype=tf.int32) # shape: (H-2,)
    # Place wall in coordinates
    obstacles_channel = tf.tensor_scatter_nd_update(obstacles_channel, wall_indices, wall) # shape: (H, W)

    # -- Poke a hole in the wall --
    # Randomly choose a location with enough room for hole and border
    hole_start = tf.random.uniform((), 1, H - 1 - hole_size, dtype=tf.int32)
    # Generate range of y coordinates for hole
    hole_y = tf.range(hole_start, hole_start + hole_size) # shape: (hole_size,)
    # Generate x coordinates for hole (a few of the wall column coordinate)
    hole_x = tf.fill([hole_size], wall_col) # shape: (hole_size,)
    # Combine into x, y pairs
    hole_indices = tf.stack([hole_y, hole_x], axis=1) # shape: (hole_size, hole_size)
    # Generate hole tensor (just a bunch of 0s)
    hole = tf.zeros((hole_size,), dtype=tf.int32) # shape: (hole_size,)
    # Place hole in coordinates
    obstacles_channel = tf.tensor_scatter_nd_update(obstacles_channel, hole_indices, hole) # shape: (H, W)


    # --- Randomly generate start and goal locations and corresponding channels ---
    # Initialize start and goal channels without committing to which is which
    left = tf.zeros_like(obstacles_channel)
    right = tf.zeros_like(obstacles_channel)

    # -- Find valid coordinates on left side of wall and place into one channel --
    # Randomly choose from any y other than borders
    left_y = tf.random.uniform((), 1, H-1, dtype=tf.int32)
    # Randomly choose from any x on left side of wall other than borders
    left_x = tf.random.uniform((), 1, wall_col, dtype=tf.int32)
    # Place 1 in chosen coordinates
    left = tf.tensor_scatter_nd_update(left, [[left_y, left_x]], [1])

    # -- Find valid coordinates on right side of wall and place into other channel --
    # Randomly choose from any y other than borders
    right_y = tf.random.uniform((), 1, H-1, dtype=tf.int32)
    # Randomly choose from any x on right side of wall other than borders
    right_x = tf.random.uniform((), wall_col+1, W-1, dtype=tf.int32)
    # Place 1 in chosen coordinates
    right = tf.tensor_scatter_nd_update(right, [[right_y, right_x]], [1])

    # -- Randomly determine which is start and which is goal --
    swap = tf.random.uniform((), 0, 2, dtype=tf.int32)
    start, goal = tf.cond(
        swap > 0,
        lambda: (right, left),
        lambda: (left, right)
    )


    # --- Stack obstacle, goal, and start channels ---
    env = tf.stack([obstacles_channel, goal, start], axis=-1) # shape: (H, W, 3)

    # --- Randomly transpose so wall is sometimes horizontal ---
    do_transpose = tf.random.uniform((), 0, 2, dtype=tf.int32)
    env = tf.cond(
        do_transpose > 0,
        lambda: tf.transpose(env, perm=[1, 0, 2]), # swap H and W
        lambda: env
    )

    return env



@tf.function
def add_goal_distance_channel(
    env: tf.Tensor,
    config: EnvConfig
) -> tf.Tensor:
    """
    Takes in obstacle, goal, start tensor and adds goal distance channel euclidean distance from each coordinate to goal.

    Parameters
    ----------
    env : tf.Tensor
        Tensor of shape (H, W, 3) with obstacle, goal, and start channels
    config : EnvConfig
        Config object specifying simulation parameters
        Attributes used: [
            idx_goal
        ]

    Returns
    ----------
    env : tf.Tensor
        Tensor of shape (H, W, 4) containing input channels + goal_distance channel
    """

    H, W, C = tf.unstack(tf.shape(env))

    # --- Identify goal coordinates ---
    goal_channel = config.idx_goal+1 # adjusted because the problem distance channel hasn't been added yet
    # Locate where goal is (where goal channel is non-zero)
    goal_pos = tf.where(env[..., goal_channel] > 0) # shape: (1, (H, W))
    # Unpack coordinates
    goal_y, goal_x = tf.unstack(tf.cast(goal_pos[0], tf.int32))

    # --- Build coordinate grid ---
    rows = tf.range(H, dtype=tf.int32)
    cols = tf.range(W, dtype=tf.int32)
    yy, xx = tf.meshgrid(rows, cols, indexing="ij") # shape: (H, W)

    # --- Compute euclidean distance from goal ---
    d2 = tf.cast((yy - goal_y) ** 2 + (xx - goal_x) ** 2, tf.float32) # shape: (H, W)
    dist = tf.sqrt(tf.cast(d2, tf.float32)) # shape: (H, W)

    # --- Add goal distance as new channel ---
    dist = tf.expand_dims(dist, axis=-1) # shape: (H, W, 1)
    # Cast full environment to floats now (previously ints)
    return tf.concat([dist, tf.cast(env, tf.float32)], axis=-1) # shape: (H, W, 4)



@tf.function
def add_problem_distance_channel(
    env: tf.Tensor,
    config: EnvConfig
) -> tf.Tensor:
    """
    Takes in obstacle, goal, start, goal_distance tensor and adds problem distance channel using value iteration to determine path lengths.

    Parameters
    ----------
    env : tf.Tensor
        Tensor of shape (H, W, 4) with obstacle, goal, start, and goal_distance channels
    config : EnvConfig
        Config object specifying simulation parameters
        Attributes used: [
            idx_goal, idx_obstacles,
            VI_step_cost, VI_goal_reward, VI_max_iters, VI_gamma, VI_theta
        ]

    Returns
    ----------
    env : tf.Tensor
        Tensor of shape (H, W, 5) containing input channels + problem_distance channel
    """

    # --- Identify goal coordinates ---
    goal_channel: int = config.idx_goal+1 # adjusted because the problem distance channel hasn't been added yet
    # Locate where goal is (where goal channel is non-zero)
    goal_pos = tf.where(env[..., goal_channel] > 0) # shape: (num_goals, 2) i.e. (1, (H, W))
    # Unpack coordinates
    goal_y, goal_x = tf.unstack(tf.cast(goal_pos[0], tf.int32))

    # --- Initialize Value and Reward tensors ---
    # Unpack shape
    H, W, _ = tf.unstack(tf.shape(env))
    # Initialize value and reward
    V = tf.zeros((H, W), dtype=tf.float32)
    R = tf.ones((H, W), dtype=tf.float32) * -config.VI_step_cost
    # Insert terminal state reward
    V = tf.tensor_scatter_nd_update(V, [[goal_y, goal_x]], [config.VI_goal_reward])
    R = tf.tensor_scatter_nd_update(R, [[goal_y, goal_x]], [config.VI_goal_reward])

    # --- Define actions ---
    actions = tf.constant([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=tf.int32)

    # Set up maze mask (1 = wall, 0 = free)
    obstacle_channel: int = config.idx_obstacles + 1 # idx adjusted because the problem distance channel hasn't been added yet
    obstacle_mask = tf.cast(env[..., obstacle_channel], tf.bool)

    # --- Setting up value iteration function for while loop ---
    def bellman_update(V, delta):

        # Pad both V and maze to avoid index problems when slicing for actions
        Vpad = tf.pad(V, [[1,1],[1,1]], constant_values=-1e9)
        Opad = tf.pad(obstacle_mask, [[1,1],[1,1]], constant_values=True)  # pad with walls


        # --- Defining transitions with fixed slices to avoid loop ---
        # Identify value of each neighbor (V(s')) for each action)
        V_up = Vpad[0:H, 1:W+1]
        V_down = Vpad[2:H+2, 1:W+1]
        V_left = Vpad[1:H+1, 0:W]
        V_right = Vpad[1:H+1, 2:W+2]
        # Identify whether neighbors are walls
        O_up = Opad[0:H, 1:W+1]
        O_down = Opad[2:H+2, 1:W+1]
        O_left = Opad[1:H+1, 0:W]
        O_right = Opad[1:H+1, 2:W+2]

        # Stack neighbors into one tensor of shape (H, W, 4)
        neighbors = tf.stack([V_up, V_down, V_left, V_right], axis=-1)
        neighbors_mask = tf.stack([O_up, O_down, O_left, O_right], axis=-1)

        # Set very negative value for transitions into walls to ensure they aren't best
        neighbors = tf.where(
            neighbors_mask,
            tf.constant(-1e9, dtype=neighbors.dtype),
            neighbors
        )

        # Find best neighbor per cell
        max_neighbor = tf.reduce_max(neighbors, axis=-1)  # shape: (H, W)

        # Updated value is reward + discounted value of best action/neighbor
        new_V = R + config.VI_gamma * max_neighbor # shape: (H, W)

        # Keep terminal value fixed
        new_V = tf.tensor_scatter_nd_update(new_V, [[goal_y, goal_x]], [config.VI_goal_reward]) # shape: (H, W)

        # Don't update wall value
        new_V = tf.where(obstacle_mask, V, new_V) # shape: (H, W)

        # Compute delta for convergence check
        delta = tf.reduce_max(tf.abs(new_V - V)) # shape: ()
        return new_V, delta

    # --- Condition for while loop ---
    # Check for convergence
    def cond(V, delta):
        return delta > config.VI_theta

    # --- Perform value iteration ---
    V_final, _ = tf.while_loop(
        cond,
        bellman_update,
        loop_vars=(V, tf.constant(1e9, tf.float32)),
        maximum_iterations=config.VI_max_iters
    )

    # --- Add new channel to environment ---
    problem_dist_channel = tf.expand_dims(V_final, axis=-1) # shape: (H, W, 1)
    return tf.concat([env, problem_dist_channel], axis=-1) # shape: (H, W, 5)



@tf.function
def generate_task(
    config: EnvConfig
) -> tf.Tensor:
    env = make_single_wall_env(
        height=config.height,
        width=config.width,
        hole_size=config.hole_size,
        seed=config.seed
    )
    env = add_goal_distance_channel(env, config)
    env = add_problem_distance_channel(env, config)
    return env




@tf.function
def generate_batch(
    config: EnvConfig
) -> tf.Tensor:
    return tf.map_fn(generate_task, tf.range(config.batch_size))