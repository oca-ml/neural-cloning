"""This script loads the cloner from a given directory
and for a series of random actions, evaluates the divergence
between the cloner and the real environment.
"""
import dataclasses
import logging

import gym
import matplotlib.pyplot as plt
import numpy as np

import cartpole.nn as cnn
import cartpole.hydra_utils as hy


LOGGER = logging.getLogger(__name__)


@hy.config
@dataclasses.dataclass
class MainConfig:
    model_path: str
    steps: int = 100


@hy.main
def main(config: MainConfig) -> None:
    cloner = cnn.NeuralCartCloner.read_model(config.model_path)

    env = gym.make("CartPole-v1")

    actions_to_take = np.random.randint(0, 2, size=config.steps)

    observation = env.reset()
    observation_0 = observation.copy()

    # 1. true environment
    obs_true = [observation]
    done_on = config.steps
    for i, action in enumerate(actions_to_take):
        observation, _reward, done, _info = env.step(action)
        if done:
            done_on = i
            break
        obs_true.append(observation)

    obs_true = np.stack(obs_true)

    # 2. simulated environment
    observation = observation_0
    obs_sim = [observation]
    for _i, action in zip(range(done_on), actions_to_take):
        net_input = cnn.wrap_net_input(state=observation, action=action)
        observation_predicted = cloner(net_input)
        observation = observation_predicted.detach().numpy()
        obs_sim.append(observation)
    obs_sim = np.stack(obs_sim)

    LOGGER.info(f"Run took {done_on} steps.")

    np.savetxt("true.csv", obs_true)
    np.savetxt("simulated.csv", obs_sim)

    fig, axs = plt.subplots(4, 1, figsize=(4, 10))
    plt.subplots_adjust(left=0.04, right=0.94, hspace=0.3)

    for i, ax in enumerate(axs):
        ax.plot(obs_true[..., i], c="k")
        ax.plot(obs_sim[..., i], c="crimson", linestyle="--")

    fig.tight_layout()
    fig.savefig("divergence.pdf")


if __name__ == "__main__":
    main()
