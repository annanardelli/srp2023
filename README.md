# Reinforcement Learning for Optimizing a Delivery Path in a Hospital Setting
## Monmouth University Summer Research Program 2023
### Anna Nardelli, Luke Shao, Brandon Hu, Dr. Jiacun Wang

### Abstract
Reinforcement learning (RL) is a type of machine learning that has many applications in real-world industry. RL intends to “teach” a model best-decision practices through exploration and trial-and-error. This project explores an application of RL in the medical field, creating a model that can efficiently navigate a space and complete tasks in a sample hospital environment. We created a simulation of a hospital floor and programmed an agent to learn the fastest route to a destination while completing tasks and avoiding obstacles. We envision our agent to be a robot that can optimally pick-up and deliver supplies/medication to specific rooms on the floor, which would make hospital practices more efficient and keep patients happier and healthier. Our model was created through a Python program and OpenAI’s Gym library. We modified a Gym environment called GridWorld, adding custom obstacles, actions, and desired locations to reach. The Python program was created to model our RL mechanism which is fundamental to the success of the project. The program creates a state-transition system using Q-learning, simulating all of the possible movement decisions the agent can make. For each of these movements, the agent receives a “reward” of some numerical value. During training, the agent is allowed to explore the environment, calculating a Q-value for each movement, which is then stored in its own matrix. The Q-value is a mathematical estimation of the immediate and long-term value of an action. The model thus learns which actions produce the highest reward through the Q-table’s analysis, and the agent learns to take the most efficient path to complete its tasks by seeking the highest possible long-term reward. Ultimately, the potential applications of this RL model are endless, with significance across industries in decision-making systems to simulate optimal outcomes.

### Winter 2023 Progress Update
The screenshot below shows a visualization of our simulation environment as of January 2024. It was adapted from OpenAI's GridWorld environment to suit the project's requirements. This grid is meant to model a hospital floor, with the green boxes being medication drop-off points and the red box being the robot's home station which it returns to after finishing its tasks.
<img width="533" alt="Screenshot 2024-01-08 at 4 21 55 PM" src="https://github.com/annanardelli/srp2023/assets/60702479/1d3abb28-4c06-4fde-847f-6ac94bcc0169">


### Future Improvements
The project will be continued during the 2023-2024 academic year by Anna Nardelli and other Monmouth students, as well as Luke Shao and Brandon Hu. The plan is to create a more complex grid and potentially implement neural networks and/or Petri Nets instead of the current state-transition system. The ultimate goal is to fully simulate a hospital floor and to run full pickup and delivery sequences which could be implemented in a real hospital.


