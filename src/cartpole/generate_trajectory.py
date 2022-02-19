"""This script can be used to "play" using either
the real environment or the cloned one.

Use:
    A to push the cart to the left
    D to push the cart to the right
    K to finish the simulation and save generated trajectory
"""
import dataclasses
from typing import List, Optional, Protocol, Tuple

import gym.envs.classic_control.cartpole as cp
import numpy as np
from pynput import keyboard

import cartpole.hydra_utils as hy
import cartpole.nn as cnn


class TransitionFunction(Protocol):
    def step(self, state: np.ndarray, action: int) -> Tuple[np.ndarray, bool]:
        ...


class RealEnv(TransitionFunction):
    def __init__(self, stop_on_done: bool = False) -> None:
        """
        Args:
            stop_on_done: whether to stop simulation on "done"

        Note:
            This may be problematic, as we don't see the cart in the screen anymore.
        """
        self.env = cp.CartPoleEnv()
        self.stop_on_done = stop_on_done

    def step(self, state: np.ndarray, action: int) -> Tuple[np.ndarray, bool]:
        self.env.state = state
        new_state, _, done, _ = self.env.step(action)
        new_state = np.asarray(new_state)

        return new_state, (self.stop_on_done & done)


class FakeEnv(TransitionFunction):
    def __init__(self, cloner: cnn.NeuralCartCloner) -> None:
        self.cloner = cloner

    def step(self, state: np.ndarray, action: int) -> Tuple[np.ndarray, bool]:
        net_input = cnn.wrap_net_input(state=state, action=action)
        output = self.cloner(net_input)

        new_state = output.detach().numpy()
        # TODO(Pawel): Teach cloner to end the game?
        return new_state, False


@hy.config
@dataclasses.dataclass
class Config:
    path: Optional[str] = None
    stop_on_done: bool = False


def get_transition_function(config: Config) -> TransitionFunction:
    if config.path is not None:
        model = cnn.NeuralCartCloner.read_model(config.path)
        return FakeEnv(model)
    else:
        return RealEnv(stop_on_done=config.stop_on_done)


def play(transition_function: TransitionFunction) -> None:
    env = cp.CartPoleEnv()
    env.reset()

    states: List[np.ndarray] = [np.asarray(env.state)]
    actions: List[int] = []

    def on_press(key):
        try:
            k = key.char
        except AttributeError:
            return
        action = 0
        if k == "a":
            action = 0
        elif k == "d":
            action = 1
        elif k == "k":
            env.close()
            return False

        actions.append(action)

        new_state, done = transition_function.step(state=states[-1], action=action)

        states.append(new_state)
        # Render new state
        env.state = new_state
        env.render()

        # If the simulator can't proceed anymore, we finish the simulation.
        if done:
            env.close()
            return False

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

    env.close()

    np.savetxt("states.csv", states)
    np.savetxt("actions.csv", actions)


@hy.main
def main(config: Config) -> None:
    transition_function = get_transition_function(config)
    play(transition_function)


if __name__ == "__main__":
    main()
