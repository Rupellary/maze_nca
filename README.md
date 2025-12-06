# Evolving Maze-Navigating NCAs

The goal of this project is to use a combination of backpropogation through time and evolutionary methods to train a cellular automata capable of navigating mazes.

A hallmark of intelligence is the ability to move further away from a goal in order to eventually reach it. Maze-navigation in which the agent can perceive the direction and distance of the goal but cannot move directly toward it makes for a simple toy model to test if NCAs can learn this rudimentary quality of intelligence. 

While NCAs were originally demonstrated as trainable with supervised machine learning, interpreting the convolution as a kind of policy allows for the application of reinforcement learning. 

## Key Challenges

### Reward-Hacking
One of the most challenging aspects of the design process is specifying a reward function that does not result in trivial behavior. With no penalties for activity, the NCA can "solve" the maze with simple space-filling behavior. But a penalty, if not carefully designed, can easily result in self-destructive dynamics becoming local minima in the loss landscape. The current design rewards the model for incremental progress towards the goal location. Progress is measured by first using a small value iteration loop to compute the number of steps each cell is from the goal location, taking the walls into account. The model is rewarded for having cells active near the goal location in "problem space" as a proportion of the total cells active. The loss for a given step is based on the delta in the reward from the previous step, with a discount on later steps in the rollout. Having the loss be based on a ratio of the activity discourages both space-filling and self-destructive behavior. Having the loss be based on the delta discourages "good-enough" behavior where the model stops before fully reaching the goal. 
The long rollouts and highly non-linear, path-dependent dynamics result (theoretically) in an extremely rugged loss landscape which traditional gradient-based exploration struggles with. Evolutionary methods were introduced in inject additional exploration into the parameter search beyond what BPTT was able to provide. Surrogate models for guided parameter sampling are being considered should the current approach suggest more directed exploration is necessary. 

### Credit Assignment
The recursive nature of the policy combined with its non-linear, path-dependent dynamics makes credit assignment very challenging for the model. A discount on later loss was used to prioritize the model's success in more probable states. Curriculum learning was also used to pre-train on tasks with fewer steps. The model is also trained on large batches of randomized tasks at a time to reduce high-frequency noise in the loss landscape. 

...