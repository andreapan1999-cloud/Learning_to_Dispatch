# sim/wrappers.py
from __future__ import annotations
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np


class DictToFlatObs(gym.ObservationWrapper):
    """
    Convert Dict obs to flat-only obs for backward compatibility.
    Expects obs["flat"] in the original dict.
    """

    def __init__(self, env: gym.Env, flat_key: str = "flat"):
        super().__init__(env)
        self.flat_key = flat_key

        # Keep exactly the same Box as the flat space
        flat_space = env.observation_space.spaces[flat_key]
        self.observation_space = flat_space

    def observation(self, observation: Dict[str, Any]) -> np.ndarray:
        return np.asarray(observation[self.flat_key], dtype=np.float32)