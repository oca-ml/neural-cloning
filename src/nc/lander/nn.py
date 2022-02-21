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
from PIL import Image


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

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16 * 5, kernel_size=5, stride=5)
        self.conv3 = nn.Conv2d(in_channels=16 * 5, out_channels=16 * 5 * 5, kernel_size=5, stride=5)

        self.full0 = nn.Linear(4, 400 * 4 * 6)

        self.conv_t1 = nn.ConvTranspose2d(in_channels=16 * 5 * 5, out_channels=16 * 5, kernel_size=5, stride=5)
        self.conv_t2 = nn.ConvTranspose2d(in_channels=16 * 5, out_channels=16, kernel_size=5, stride=5)
        self.conv_t3 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=4, stride=4)

    def forward(self, observation, action):
        x = observation
        x0 = x.permute(2, 0, 1)[None, ...]

        # Downsampling
        x1 = self.conv1(x0)
        x1 = F.relu(x1)

        x2 = self.conv2(x1)
        x2 = F.relu(x2)

        x3 = self.conv3(x2)
        x3 = F.relu(x3)

        # Bottleneck layer
        y = self.full0(action).reshape(*x3.shape)
        mid = x3 + y

        # Upsampling
        xt1 = self.conv_t1(mid) + x2
        xt1 = F.relu(xt1)

        xt2 = self.conv_t2(xt1) + x1
        xt2 = F.relu(xt2)

        xt3 = self.conv_t3(xt2) + x0

        # Return
        out = xt3[0].permute(1, 2, 0)
        return out

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
    env = env or gym.make("LunarLander-v2")
    optimizer = optimizer_factory(cloner.parameters())
    if policy is not None:
        raise NotImplementedError("Non-random policy not allowed.")

    device = torch.device(device)

    cloner = cloner.to(device)

    resets = []
    losses = []
    state = env.reset()
    observation = env.render(mode="rgb_array")
    observation_tensor = torch.tensor(observation, dtype=torch.float) / 255 - 0.5

    t0 = time.time()

    for i in range(1, n_steps + 1):
        if i % (n_steps // 100) == 0:
            delta_t = time.time() - t0
            message = f"{100*i/n_steps:.1f}%, loss: {np.mean(losses[:-100]):.2e}, time: {delta_t:.1f}"
            LOGGER.info(message)

            image_arr = observation_predicted.cpu().detach().numpy()  # noqa
            image_arr = np.clip(image_arr, -0.5, 0.5)
            img = Image.fromarray(np.uint8(255 * (image_arr + 0.5)))

            dir = pathlib.Path("images")
            dir.mkdir(exist_ok=True)

            img.save(dir / f"{i:5d}-2-generated.png")
            Image.fromarray(observation).save(dir / f"{i:5d}-1-true.png")

        # TODO(Pawel): Substitute for non-random policy.
        action = env.action_space.sample()

        action_tensor = torch.tensor(np.eye(4)[action], dtype=torch.float)

        observation_predicted = cloner(observation=observation_tensor.to(device), action=action_tensor.to(device))
        state, _reward, done, _info = env.step(action)
        observation = env.render(mode="rgb_array")

        if done:
            LOGGER.warning(f"At step {i} we needed to reset the environment.")
            resets.append(i)
            state = env.reset()  # noqa
            observation = env.render(mode="rgb_array")
            continue

        # We optimize for the mean sum-of-squared loss
        observation_tensor = torch.tensor(observation, dtype=torch.float).to(device) / 255 - 0.5

        loss = ((observation_tensor - observation_predicted) ** 2).mean()

        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    env.close()

    LOGGER.info(f"Training finished. Loss: {np.mean(losses[:-100]):.2e}. Total time: {time.time() - t0:.1f}.")
    return ResultTuple(cloner=cloner, losses=losses, resets=resets)
