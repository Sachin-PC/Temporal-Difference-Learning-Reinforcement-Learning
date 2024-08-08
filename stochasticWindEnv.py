from enum import IntEnum
from typing import Tuple, Optional, List
from gym import Env, spaces
from gym.utils import seeding
from gym.envs.registration import register
import numpy as np


def register_env() -> None:
    """Register custom gym environment so that we can use `gym.make()`

    In your main file, call this function before using `gym.make()` to use the Four Rooms environment.
        register_env()
        env = gym.make('WindyGridWorld-v0')

    There are a couple of ways to create Gym environments of the different variants of Windy Grid World.
    1. Create separate classes for each env and register each env separately.
    2. Create one class that has flags for each variant and register each env separately.

        Example:
        (Original)     register(id="WindyGridWorld-v0", entry_point="env:WindyGridWorldEnv")
        (King's moves) register(id="WindyGridWorldKings-v0", entry_point="env:WindyGridWorldEnv", **kwargs)

        The kwargs will be passed to the entry_point class.

    3. Create one class that has flags for each variant and register env once. You can then call gym.make using kwargs.

        Example:
        (Original)     gym.make("WindyGridWorld-v0")
        (King's moves) gym.make("WindyGridWorld-v0", **kwargs)

        The kwargs will be passed to the __init__() function.

    Choose whichever method you like.
    """
    # TODO


class Action(IntEnum):
    """Action"""

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
    LEFT_UP = 4
    LEFT_DOWN = 5
    RIGHT_UP = 6
    RIGHT_DOWN = 7


def actions_to_dxdy(action: Action) -> Tuple[int, int]:
    """
    Helper function to map action to changes in x and y coordinates
    Args:
        action (Action): taken action
    Returns:
        dxdy (Tuple[int, int]): Change in x and y coordinates
    """
    mapping = {
        Action.LEFT: (-1, 0),
        Action.DOWN: (0, -1),
        Action.RIGHT: (1, 0),
        Action.UP: (0, 1),
        Action.LEFT_UP: (-1,1),
        Action.LEFT_DOWN: (-1,-1),
        Action.RIGHT_UP: (1,1),
        Action.RIGHT_DOWN: (1,-1),
    }
    return mapping[action]


class StochasticWindyGridWorldEnv(Env):
    def __init__(self):
        """Windy grid world gym environment
        This is the template for Q4a. You can use this class or modify it to create the variants for parts c and d.
        """

        # Grid dimensions (x, y)
        self.rows = 10
        self.cols = 7

        # Wind
        # TODO define self.wind as either a dict (keys would be states) or multidimensional array (states correspond to indices)
        self.wind = {}
        for r in range(self.rows):
            for c in range(self.cols):
                state = (r,c)
                if r in [3,4,5,8]:
                    self.wind[state] = 1
                elif r in [6,7]:
                    self.wind[state] = 2
                else:
                    self.wind[state] = 0
        print(self.wind)

        self.action_space = spaces.Discrete(len(Action))
        print("self.action_space")
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(self.rows), spaces.Discrete(self.cols))
        )

        # Set start_pos and goal_pos
        self.start_pos = (0, 3)
        self.goal_pos = (7, 3)
        self.agent_pos = None

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Fix seed of environment

        In order to make the environment completely reproducible, call this function and seed the action space as well.
            env = gym.make(...)
            env.seed(seed)
            env.action_space.seed(seed)

        This function does not need to be used for this assignment, it is given only for reference.
        """

        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.agent_pos = self.start_pos
        return self.agent_pos

    def stepOld(self, action: Action) -> Tuple[Tuple[int, int], float, bool, dict]:
        """Take one step in the environment.

        Takes in an action and returns the (next state, reward, done, info).
        See https://github.com/openai/gym/blob/master/gym/core.py#L42-L58 foand r more info.

        Args:
            action (Action): an action provided by the agent

        Returns:
            observation (object): agent's observation after taking one step in environment (this would be the next state s')
            reward (float) : reward for this transition
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning). Not used in this assignment.
        """

        # TODO
        reward = None
        done = None

        return self.agent_pos, reward, done, {}

    def step(self, action: Action) -> Tuple[Tuple[int, int], float, bool, dict]:
        """Take one step in the environment.

        Takes in an action and returns the (next state, reward, done, info).
        See https://github.com/openai/gym/blob/master/gym/core.py#L42-L58 foand r more info.

        Args:
            action (Action): an action provided by the agent

        Returns:
            observation (object): agent's observation after taking one step in environment (this would be the next state s')
            reward (float) : reward for this transition
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning). Not used in this assignment.
        """

        # Check if goal was reached
        if self.agent_pos == self.goal_pos:
            done = True
            reward = 0.0
        else:
            done = False
            reward = -1.0
        
        action_taken = action

        # TODO modify action_taken so that 10% of the time, the action_taken is perpendicular to action (there are 2 perpendicular actions for each action).
        # You can reuse your code from ex0
        # action_taken = action
        # probablity_value = random.uniform(0, 1)
        # if(probablity_value <= 0.9):
        #     action_taken = action
        # elif probablity_value <= 0.95:
        #     if action == Action.RIGHT or action == Action.LEFT:
        #         action_taken = Action.UP
        #     else:
        #         action_taken = Action.RIGHT
        # else:
        #     if action == Action.RIGHT or action == Action.LEFT:
        #         action_taken = Action.DOWN
        #     else:
        #         action_taken = Action.LEFT

        # TODO calculate the next position using actions_to_dxdy()
        # You can reuse your code from ex0
        next_pos = None
        # print("Print action taken  = ",action_taken)
        next_pos_mapping = actions_to_dxdy(action_taken)
        # print("Self.agent_pos = ",self.agent_pos," action = ",action," next_pos_mapping before wind= ",next_pos_mapping)
        next_pos_mapping = (next_pos_mapping[0], next_pos_mapping[1] + self.wind[self.agent_pos])
        if self.wind[self.agent_pos] in [1,2]:
            prob_val = np.random.rand()
            if prob_val < (1.0/3.0):
                next_pos_mapping = (next_pos_mapping[0], next_pos_mapping[1] + 1)
            elif prob_val < (2.0/3.0):
                next_pos_mapping = (next_pos_mapping[0], next_pos_mapping[1] - 1)
        # print("Self.agent_pos = ",self.agent_pos," action = ",action," next_pos_mapping after wind= ",next_pos_mapping)
        next_pos = tuple(map(lambda i, j: i + j, self.agent_pos, next_pos_mapping))
        # if(next_pos == (10,10)):
            # reward = 1
        next_pos_x = max(0,next_pos[0])
        next_pos_x = min(9,next_pos_x)
        next_pos_y = max(0,next_pos[1])
        next_pos_y = min(6,next_pos_y)
        next_pos = (next_pos_x,next_pos_y) 
        # if next_pos[0] <0 or next_pos[0] > 9 or next_pos[1] < 0 or next_pos[1] > 6:
        #     next_pos = self.agent_pos

        self.agent_pos = next_pos
        # print(next_pos)
        if next_pos == self.goal_pos:
            reward = 0
            done = True

        # TODO check if next position is feasible
        # If the next position is a wall or out of bounds, stay at current position
        # Set self.agent_pos
        # self.agent_pos = next_pos

        return self.agent_pos, reward, done, {}
