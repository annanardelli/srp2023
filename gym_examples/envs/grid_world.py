# Code for producing grid world environment

import gym
from gym import spaces
import pygame
import numpy as np


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 2}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(5)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, -1]),
            2: np.array([-1, 0]),
            3: np.array([0, 1]),
            4: np.array([0, 0]) #Pick up
        }

        self.is_trained = False

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def get_state_size(self):
        return self.size * self.size * len(self._med_locations) *2
    def get_size(self):
        return self.size

    def get_states(self):
        states = {}
        index = 0
        picked_up = np.array([False] * len(self._med_locations))
        #for n in range(len(self._med_locations)):
        counter = 0
        for t in range(2):
            picked_up[1] = False
            if t == 1:
                picked_up[1] = True
            for z in range(2):
                picked_up[0] = False
                if z == 1:
                    picked_up[0] = True
                for x in range(self.size):
                    for y in range(self.size):
                        tpl = tuple(picked_up)
                        pair = ((y, x), tpl)
                        states.update({pair: index})
                        index = index + 1
        return states

    def get_is_picked_up(self):
        return tuple(self.is_picked_up)

    def trained(self):
        self.is_trained = True
        input("Press Enter to watch trained agent...")

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Set target location (stationary)
        self._target_location = np.array([self.size - 1, self.size - 1])

        # Choose the agent's location uniformly at random
        """
        self._agent_location = self._target_location
        while np.array_equal(self._agent_location, self._target_location):
            self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        """
        # Same place agent location reset
        self._agent_location = np.array([0,0])


        # Random target location reset
        """
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int)
            """
        # Med location
        self._med_locations = np.array([[0,self.size-1], [self.size-1,0]])
        self.is_picked_up = np.array([False]*len(self._med_locations))

        # obstacle location
        self._obs_locations = np.array([[0,self.size-3], [self.size-3,0]])
        """
        self._obs_locations = np.array([self._target_location])
        while np.array_equal(self._obs_locations[0], self._target_location) or np.array_equal(self._obs_locations[0], self._agent_location):
            self._obs_locations[0] = self.np_random.integers(0, self.size, size=2, dtype=int)
        """

        observation = self._get_obs()
        info = self._get_info()

        if self.is_trained:
            if self.render_mode == "human":
                self._render_frame()
            elif self.render_mode == "rgb_array":
                self.render()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        print(f"Grid Action: {action}")

        direction = self._action_to_direction[action]

        observation = self._get_obs()  # Gets original/current state
        # Indexes current state
        states = self.get_states()
        pairTuple = (tuple(observation["agent"]), self.get_is_picked_up())
        state = states[pairTuple]
        print(f"Grid State: {state}")

        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # An episode is done if the agent has dropped off the med and reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)

        observation = self._get_obs()  # new state
        info = self._get_info()

        """
        # rewards
        rewards_matrix = [[5,-10,-10,5],
                          [0,-10,-5,5],
                          [0,-10,-5,5],
                          [0,-10,-5,5],
                          [-10,-10,-5,5],
                          [5,-5,-10,0],
                          [5,-5,-5,5],
                          [0,-5,-5,-1000],
                          [0,-5,-5,5],
                          [-10,-5,-5,5],
                          [5,-5,-10,0],
                          [-1000,-5,-5,0],
                          [5,-5,-5,5],
                          [0,-5,-1000,5],
                          [-10,-5,-5,5],
                          [5,-5,-10,5],
                          [5,-5,-5,5],
                          [5,-1000,-5,5],
                          [5,-5,-5,5],
                          [-10,-5,-5,5],
                          [5,-5,-10,-10],
                          [5,-5,-5,-10],
                          [5,-5,-5,-10],
                          [5,-5,-5,-10],
                          [-10,-5,-5,-10]]
        reward = 100 if terminated else rewards_matrix[state][action]  #rewards_matrix
        """
        # Binary sparse rewards
        reward = -1
        if terminated:
            reward = 100
        for x in range(len(self._med_locations)):
            if action == 4 and np.array_equal(self._med_locations[x], self._agent_location) and not self.is_picked_up[x]:
                self.is_picked_up[x] = True
                reward = 1000

        for x in range(len(self._obs_locations)):
            if np.array_equal(self._obs_locations[x], self._agent_location):
                reward = -1000

        if self.is_trained:
            if self.render_mode == "human":
                self._render_frame()
            elif self.render_mode == "rgb_array":
                self.render()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Obstacles
        for x in range(len(self._obs_locations)):
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(
                    pix_square_size * self._obs_locations[x],
                    (pix_square_size, pix_square_size),
                ),
            )
        #Pick up location
        for x in range(len(self._med_locations)):
            pygame.draw.rect(
                canvas,
                (0, 255, 0),
                pygame.Rect(
                    pix_square_size * self._med_locations[x],
                    (pix_square_size, pix_square_size),
                ),
            )
            y = x + 1
            if self.is_picked_up[x] == False:
                pygame.draw.circle(
                    canvas,
                    (175, 30*y, 51*y),
                    (self._med_locations[x] + 0.5) * pix_square_size,
                    pix_square_size / (4+2*x),
                )

        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Meds picked up
        for x in range(len(self._med_locations)):
            if self.is_picked_up[x]:
                y = x + 1
                pygame.draw.circle(
                    canvas,
                    (175, 30*y, 51*y),
                    (self._agent_location + 0.5) * pix_square_size,
                    pix_square_size / (4+2*x),
                )



        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
