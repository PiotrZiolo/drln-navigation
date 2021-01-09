# Project 1: Navigation

### Project description

This project solves the Banana environment from the Unity engine. The goal was to train an agent to navigate and collect yellow bananas while avoiding blue bananas in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

At each time step, the agent has four actions at its disposal:
- `0` - walk forward 
- `1` - walk backward
- `2` - turn left
- `3` - turn right

The state space has `37` dimensions. The first 35 dimensions contain ray-based perception of objects around agent's forward direction. The before last variable contains the rotation speed of the agent and the last variable contains the forward/backward speed of the agent. 

There are 7 rays used, each encoded in 5 numbers. The rays go out of the agent at the following angles [20, 90, 160, 45, 135, 70, 110], where 90 degrees is the direction exactly ahead of the agent. The first four variables in the state vector describing  a given ray indicate if a given type of object from the list [Banana, Wall, BadBanana, Agent, Distance] (in that order) is present along the ray. The fifth variable gives the distance to the closest object along the ray. For instance [1, 1, 0, 0, 0.5] means that there is a yellow banana and a wall along the ray and that the banana is halfway along the ray length. (The exact logic of rays was figured out from the source code by iandanforth - https://github.com/Unity-Technologies/ml-agents/issues/1134)

The environment is solved when the agent achieves an average score of +13 over 100 consecutive episodes.

### Requirements to run the agent

1.  Create (and activate) a new environment with Python 3.6 with conda.

    -   **Linux** or **Mac**:

    <!-- -->

        conda create --name drlnavigation python=3.6
        source activate drlnavigation

    -   **Windows**:

    <!-- -->

        conda create --name drlnavigation python=3.6 
        activate drlnavigation

2.  Install OpenAI gym.

        pip install gym

3.  Clone or copy just `python` folder from the following
    [repo](https://github.com/udacity/deep-reinforcement-learning)
    inside this repo, then change directory to the copied folder and
    install it with the following command.

        pip install .

4. Download the Unity Banana Collector environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

5. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

6. Start Jupyter Notebook and open Banana_navigation_solution.ipynb. This notebook allows to train two agents (Double DQN, Prioritized Double DQN) and run a test of the trained agents. The code of the agents is containted in the files:

	- ddqn_agent.py,
	- prioritized_ddqn_agent.py,
	- model.py (both agents use the same network architecture).
	