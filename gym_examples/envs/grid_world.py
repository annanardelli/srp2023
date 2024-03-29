# Code for producing grid world environment

import gym
from gym import spaces
import pygame
import numpy as np
import itertools


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
            4: np.array([0, 0])  # Pick up
        }

        # Initial trained status
        self.is_trained = False

        self._obs_locations = None

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
        return self.size * self.size * (2 ** len(self._med_locations))

    def get_size(self):
        return self.size

    def get_states(self):
        states = {}
        index = 0

        staterange = range(self.size)
        for c in itertools.product(staterange, repeat=2):
            for p in itertools.product([True, False], repeat=len(self._med_locations)):
                    pair = (c, p)
                    states[pair] = index
                    index += 1

        print(states)
        return states

    def get_is_picked_up(self):
        return tuple(self.is_picked_up)

    def trained(self):
        self.is_trained = True
        input("Press Enter to watch trained agent...")

    def set_obstacles(self, x1, y1, x2, y2):
        self._obs_locations = np.array([[x1, y1], [x2, y2]])

    def set_obstacles_complex(self):
        self._obs_locations = np.array([[2, 1], [2, 2], [2, 3], [2, 4], [1, 4], [0, 4],
                                        [5, 0], [5, 1], [5, 2], [5, 4], [6, 4], [7, 4], [8, 4], [9, 4],
                                        [0, 7], [1, 7], [3, 7], [3, 8], [3, 9]])

    def set_obstacles_hospital(self):
        self._obs_locations = np.array([[0, 0], [0, 9], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6],
                                        [3, 2], [3, 3], [3, 4], [3, 5], [3, 6],
                                        [4, 2], [4, 5], [4, 6],
                                        [6, 2], [6, 3], [6, 5], [6, 6],
                                        [7, 2], [7, 3], [7, 5], [7, 6],
                                        [8, 2], [8, 3],
                                        [9, 0], [9, 2], [9, 3], [9, 9]])

    def get_obstacles(self):
        return self._obs_locations

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
        self._target_location = np.array([4, 4])

        # Set agent location (stationary)
        self._agent_location = np.array([4, 3])
        self._prev_location = []

        # Med location
        self._med_locations = np.array([[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6]])

        '''
        , [0,7], [0,8],
        [1,0], [2,0], [3,0], [4,0], [5,0], [6,0], [7,0], [8,0],
        [9,1], [9,4], [9,5], [9,6], [9,7], [9,8],
        [1,9], [2,9], [3,9], [4,9], [5,9], [6,9], [7,9], [8,9]'''

        self.is_picked_up = np.array([False] * len(self._med_locations))

        # Obstacle location
        self._obs_locations = self.get_obstacles()

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
        direction = self._action_to_direction[action]

        observation = self._get_obs()  # Gets original/current state
        # Indexes current state

        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # An episode is terminated if the agent has dropped off the medicine and reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)

        observation = self._get_obs()  # New state
        info = self._get_info()

        # BINARY SPARSE REWARDS
        reward = -1
        # Reward for terminating at the ending location
        if terminated:
            reward = 100
        for x in range(len(self._med_locations)):
            # Penalty for moving between med pick up locations (rooms)
            if np.array_equal(self._prev_location, self._med_locations[x]):
                for y in range(len(self._med_locations)):
                    if np.array_equal(self._med_locations[y], self._agent_location):
                        reward = -500
            # Reward for correct med location pick up
            if action == 4 and np.array_equal(self._med_locations[x], self._agent_location) and not self.is_picked_up[
                x]:
                self.is_picked_up[x] = True
                reward = 1000

        # Penalty for moving into an obstacle
        for x in range(len(self._obs_locations)):
            if np.array_equal(self._obs_locations[x], self._agent_location):
                reward = -1000

        self._prev_location = observation["agent"]

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

        # DRAWING GAMEBOARD
        # Target
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
        # Pick up location
        for x in range(len(self._med_locations)):
            pygame.draw.rect(
                canvas,
                (0, 255, 0),
                pygame.Rect(
                    pix_square_size * self._med_locations[x],
                    (pix_square_size, pix_square_size),
                ),
            )

            if self.is_picked_up[x] == False:
                pygame.draw.circle(
                    canvas,
                    (205, 255, 255),
                    (self._med_locations[x] + 0.5) * pix_square_size,
                    pix_square_size / 4,
                )

        # Agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Meds to be picked up
        for x in range(len(self._med_locations)):
            if self.is_picked_up[x]:
                y = x + 1
                pygame.draw.circle(
                    canvas,
                    (205, 255, 255),
                    (self._agent_location + 0.5) * pix_square_size,
                    pix_square_size / (4 + x),
                )

        # Gridlines
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
