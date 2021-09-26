
# Project 3: Tennis: Collaboration and Competition

### Introduction

This report describes the reinforcement learning agent designed to solve the Unity Tennis environment. Solving the environment entails averaging a max score of 0.5 or more between both agent for at least 100 consecutive episodes. The environment dynamics are described by Markov Descision Process with states, actions, and rewards detailed in the ***Environment*** section. The reinforcement learning algorithm used to solve the environment is described the ***RL Agent*** section. The model architectures that define the Actor/Critic networks are explained in the ***Model Architecture*** section. Discussion of the agent' final results is located in the ***Final Performance*** section

### Environment

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Therefore, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. An agent's full observation is comprised of 24 variables, which corresponds to 3 sequential local observations concatenated together. Each agent receives its own, full local observation.

Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic. In order to solve the environment, the DRL agent must get an average score of +0.5 over 100 consecutive episodes, where the average is taken using the maximum score between agents each episode. Specifically,

* After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
* This yields a single score for each episode.

A set of state, action, reward, and next state values together comprise an experience tuple

### RL Agent

A Deep Deterministic Policy Gradient (DDPG) reinforcement learning agent was chosen to solve this environment for its applicability to continous control problems. A DDPG agent features an Actor/Critic framework in which the Actor deep neural network, which approximates the optimal policy, determines the actions through which the agent interacts with the environment and the Critic deep neural network, which approximates the optimal action-value function, judges the quality of the selected actions.

There are two instances of each Actor/Critic network - a local and a target network - for a total of four networks internal to the agent. The target networks serve as the reference for the loss signal and maintain a minimally changing set of weights throughtout the learning process. The local networks perform learning through error backpropagation and update weights more frequently. At a user-specified interval, the target network weights are updated to more closely match the local network to move the target reference in the direction of learning. Initially, the local and target network weights start with the same weights.

At each time step of an episode, the agent performs an action to generate an experience tuple that it adds to a memory buffer. After accumulating enough experiences, the agent randomly samples a batch of experiences from the memory buffer to generate a learning signal by comparing expected and target Q values that are output from forward passes through the local and target networks. The backpropagated loss signal is derived from the difference between these two estimates. Throughout the learning steps, the agent also performs update steps to copy the new local network weights to the target network. The agent converges on the optimal policy that maximizes the expected cumulative reward by repeating this process over many episodes.

The agent's reward dynamics are driven by a set of user-defined hyperparameters:
- BUFFER_SIZE: the maximum number of experience tuples the agent can store
- BATCH_SIZE: the number of experiences the agent sample each learning step
- GAMMA: discount factor of future rewards
- TAU: amount that target networks update to match local networks
- LR_ACTOR: learning rate of the actor network
- LR_CRITIC: learning rate of critic network
- WEIGHT_DECAY: factor that drives how the optimization algorithm changes weights
- LEARN_EVERY: number of time steps for every learning step
- NUM_LEARNS: how many times to learn at each learn step
- UPDATE_EVERY: number of learning steps for every target network update step
- NUM_UPDATES: number of target network updates for every update step
- NOISE_VAR: scale factor applied to variance of action space noise
- NOISE_REDUCTION: factor to reduce action space noise over training duration
  
### Model Architecture

All four deep neural networks are comprised of two fully connected hidden layers and an output layer. The Actor networks' output layer size corresponds to the dimensionality of the action space. The Critic networks' output layer size is 1, which represents the scalar state-action value. Each network also utilized a one-dimensional batch normalization layer after the first fully connected layer to stabalize the network statistics. All hidden layers used rectified linear unit activation functions. The Actor network applied a hyperbolic tangent function to the output layer logits to ensure that values remained within a +/- 1 domain. The Critic network output layer returned the logits without any applied activation function.

### Final Performance

The DRL agent exhibited slow learning during the initial several hundred episodes, then quickly progressed to achieve the passing score.

Over the course of developing the DRL algorithm, the agent's performace demonstrated one characteristic that was both surprising and concerning - ***sensitivity to methodology of class definitions***. Keeping the same hyperparameters values and pseudo-random number generator seed, the agent required wildly different numbers of episodes to solve the environment just by changing how classes were defined and how they received inputs. One training session with all classes defined in the Jupyter Notebook produced one result, and one training session defining classes in separate Python files produced another. Changing network parameters such as state_size and action_size from globals to function inputs produced yet another. All these methods of effectively defining the same architecture with the same constituent values produced solutions ranging from 409 episodes to 1618 episodes.





