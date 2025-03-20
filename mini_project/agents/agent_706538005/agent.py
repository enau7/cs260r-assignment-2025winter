import pathlib

import torch
import numpy as np

import sys
import pathlib

current_dir = pathlib.Path(__file__).parent
sys.path.insert(0, str(current_dir))

from stable_baselines3.ppo import PPO
from stable_baselines3.ppo.policies import ActorCriticPolicy

class Policy:
    """
    This class is the interface where the evaluation scripts communicate with your trained agent.

    You can initialize your model and load weights in the __init__ function. At each environment interactions,
    the batched observation `obs`, a numpy array with shape (Batch Size, Obs Dim), will be passed into the __call__
    function. You need to generate the action, a numpy array with shape (Batch Size, Act Dim=2), and return it.

    Do not change the name of this class.

    Please do not import any external package.
    """
    # FILLED YOUR PREFERRED NAME & UID HERE!
    CREATOR_NAME = "Colton Rowe"  # Your preferred name here in a string
    CREATOR_UID = "706538005"  # Your UID here in a string

    def __init__(self):
        model_path = pathlib.Path(__file__).parent / "model.zip"
        # model_path = r"C:\Users\Colton\Documents\GitHub\cs260r-assignment-2025winter\mini_project\runs\ppo_metadrive_500_scenarios\ppo_metadrive_500_scenarios_2025-03-19_14-27-44_7a2b10a5\models\rl_model_1000000_steps.zip"
        self.agent = PPO.load(model_path)

    def __call__(self, obs):
        obs = np.array(obs, dtype=np.float32)
        actions, _ = self.agent.predict(obs, deterministic=False)
        return actions
