# Evolving Maze-Navigating NCAs

The goal of this project is to use a combination of backpropogation through time and evolutionary methods to train a cellular automata capable of navigating mazes.

A hallmark of intelligence is the ability to move further away from a goal in order to eventually reach it. Maze-navigation in which the agent can perceive the direction and distance of the goal but cannot move directly toward it makes for a simple toy model to test if NCAs can learn this rudimentary quality of intelligence. 

While NCAs were originally demonstrated as trainable with supervised machine learning, interpreting the convolution as a kind of policy allows for the application of reinforcement learning. 

## Key Challenges

### Reward-Hacking
One of the most challenging aspects of the design process is specifying a reward function that does not result in trivial behavior. With no penalties for activity, the NCA can "solve" the maze with simple space-filling behavior. But a penalty, if not carefully designed, can easily result inself-destructive dynamics becoming local minima. 

### Credit Assignment
The recursive nature of the policy combined with its extremely non-ergodic dynamics makes credit assignment extremely challenging for the model. A discount on later loss was used to prioritize the model's success in more probable states. Curriculum learning was also used to pre-train on tasks with fewer steps.

...