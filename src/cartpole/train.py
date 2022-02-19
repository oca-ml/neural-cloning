"""This script trains the cloner."""
import dataclasses
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

import cartpole.nn as cp
import cartpole.hydra_utils as hy


@dataclasses.dataclass
class PlotConfig:
    """
    Attrs:
        smoothing_window: used to smooth out the loss curve
        reset: reset times are marked with ticks. This controls the tick length.
    """

    smoothing_window: float = 0.05
    reset: float = 0.1


@hy.config
@dataclasses.dataclass
class MainConfig:
    """
    Attrs:
        device: device to be used. Beware: CUDA is untested.
        n_steps: training length
        middle: size of the middle layer in cloner
    """

    device: Optional[str] = "cpu"
    n_steps: int = 10_000
    middle: int = 4
    plot: PlotConfig = dataclasses.field(default_factory=PlotConfig)


def smooth_loss(window: float, losses: Sequence[float]) -> np.ndarray:
    window_len = int(len(losses) * window + 1)
    kernel = np.ones(window_len) / window_len

    return np.convolve(losses, kernel, mode="valid")


def plot_training(result: cp.ResultTuple, ax: plt.Axes, config: PlotConfig) -> None:
    loss_smooth = smooth_loss(window=config.smoothing_window, losses=result.losses)

    ax.plot(loss_smooth, c="k")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")

    ymin, ymax = np.min(loss_smooth), np.max(loss_smooth)
    ax.vlines(result.resets, ymin=ymax / (ymax / ymin) ** config.reset, ymax=ymax, colors="r")


@hy.main
def main(config: MainConfig) -> None:
    result = cp.train(
        cloner=cp.NeuralCartCloner(middle=config.middle),
        n_steps=config.n_steps,
        device=config.device,
    )

    # Save the model. We use the fact that hydra substituted the directory.
    result.cloner.save_model(".")

    # Plot training curve
    fig, ax = plt.subplots()

    plot_training(
        result=result,
        ax=ax,
        config=config.plot,
    )

    fig.tight_layout()
    fig.savefig("loss.pdf")


if __name__ == "__main__":
    main()
