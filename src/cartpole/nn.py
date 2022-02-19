import pathlib
import time
import dataclasses
import json
import logging
from typing import List, Union

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOGGER = logging.getLogger(__name__)


class NeuralCartCloner(nn.Module):
    _state_dict_path: str = "cloner_state_dict.py"
    _hyperparams_path: str = "cloner_hyperparams.json"

    def __init__(self, middle: int = 4) -> None:
        """Neural network modelling the transition function.

        Signature:
            (State, Action) -> State

        In our case, it maps a (4+2)-dim vector vector into 4-dim vector.
        """
        super().__init__()
        self._middle = middle

        self.full0 = nn.Linear(6, middle)
        self.full1 = nn.Linear(middle, 4)

    def forward(self, x):
        y = F.relu(self.full0(x))
        y = self.full1(y)
        return x[:4] + y

    def save_model(self, path: Union[str, pathlib.Path] = ".") -> None:
        path = pathlib.Path(path)

        path_state_dict = path / self._state_dict_path
        torch.save(self.state_dict(), path_state_dict)

        with open(path / self._hyperparams_path, "w") as f:
            json.dump(
                {
                    "middle": self._middle,
                },
                fp=f,
            )

    @classmethod
    def read_model(cls, path: Union[str, pathlib.Path]) -> "NeuralCartCloner":
        path = pathlib.Path(path)

        with open(path / cls._hyperparams_path) as f:
            kwargs = json.load(f)

        model = cls(**kwargs)
        model.load_state_dict(torch.load(path / cls._state_dict_path))
        model.eval()
        return model


def _net_input(state, action: int) -> torch.Tensor:
    """Wraps the state (array/list/tuple with 4 components)
    and the action (0 or 1) into a tensor of shape (6,).

    The first four components are the state
    and the last two are one-hot-encoded action.
    """
    net_input = np.zeros(6)
    net_input[:4] = state
    net_input[4 + action] = 1
    return torch.tensor(net_input, dtype=torch.float)


@dataclasses.dataclass
class ResultTuple:
    """
    Attrs:
        cloner: trained cloner
        losses: list of losses during the training
        resets: training steps at which the environment needed to be reset
    """

    cloner: nn.Module
    losses: List[float]
    resets: List[int]


def train(
    cloner: nn.Module,
    env=None,
    optimizer_factory=optim.Adam,
    n_steps: int = 10_000,
    policy=None,
    device: str = DEVICE,
) -> ResultTuple:
    """Trains the cloner.

    Args:
        cloner: neural network used to clone the environment
        env: cart-pole environment, defaults to CartPole-v1
        optimizer_factory: a function which takes nn's parameters
            and returns an optimizer. Defaults to Adam with default parameters.
        n_steps: length of the training
        policy: policy used to generate trajectories for environment cloning.
            Defaults to the random policy.
        device: device used to train the cloner
    """
    # Fill in missing values
    env = env or gym.make("CartPole-v1")
    optimizer = optimizer_factory(cloner.parameters())
    if policy is not None:
        raise NotImplementedError("Non-random policy not allowed.")

    device = torch.device(device)

    cloner = cloner.to(device)

    resets = []
    losses = []
    observation = env.reset()

    t0 = time.time()

    for i in range(1, n_steps + 1):
        if i % (n_steps // 100) == 0:
            delta_t = time.time() - t0
            message = f"{100*i/n_steps:.1f}%, loss: {np.mean(losses[:-100]):.2e}, time: {delta_t:.1f}"
            LOGGER.info(message)
            # print(f"\r{message}", end="")

        # TODO(Pawel): Substitute for non-random policy.
        action = env.action_space.sample()

        net_input = _net_input(state=observation, action=action).to(device)

        observation_predicted = cloner(net_input)
        observation, _reward, done, _info = env.step(action)

        if done:
            LOGGER.warning(f"At step {i} we needed to reset the environment. The observation was: {observation}.")
            resets.append(i)
            observation = env.reset()
            continue

        # We optimize for the mean sum-of-squared loss
        observation_tensor = torch.tensor(observation, dtype=torch.float).to(device)

        loss = ((observation_tensor - observation_predicted) ** 2).sum()

        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    env.close()

    LOGGER.info(f"Training finished. Loss: {np.mean(losses[:-100]):.2e}. Total time: {time.time() - t0:.1f}.")
    return ResultTuple(cloner=cloner, losses=losses, resets=resets)
