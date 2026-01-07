import os
from datetime import datetime, timezone
from pathlib import Path

import torch
import numpy as np
import pandas as pd  # type: ignore
from matplotlib import pyplot as plt


def dt_str() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def log(m: str, log_type: str = "INFO") -> None:
    print(f"{dt_str()} {log_type}: {m}")
    return None


def wd() -> Path:
    """
    Set and return the working directory

    :return: the working directory
    """
    fp = Path(__file__).resolve()
    wd = fp.parent.parent
    os.chdir(wd)
    return wd


def get_available_device() -> torch.device:
    """
    Return the best available PyTorch device:
    - CUDA if available
    - else MPS if available (macOS Metal backend)
    - else CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def plot_metrics(dfs: tuple[pd.DataFrame, pd.DataFrame], fp: Path):
    fig, ax1 = plt.subplots(figsize=(10, 8))
    # ax2 = ax1.twinx() # a second axis for accuracy

    for df in dfs:
        assert "step" in df.columns, 'We need a "step" column'
        x = df["step"].values
        for col in df.columns:
            if col == "step":
                continue

            y = df[col].values
            if "loss" in col:
                ax1.plot(
                    x,  # type: ignore
                    y,  # type: ignore
                    label=col.replace("_", " ").title(),
                    alpha=(0.5 if "train" in col else None),
                )  # type: ignore
            # elif "accura" in col:
            #     ax2.plot(
            #         x, # type: ignore
            #         y, # type: ignore
            #         label=col.replace("_", " ").title(),
            #         alpha=(0.5 if "train" in col else None)
            #     ) # type: ignore

    ax1.set_title("Train & Test Metrics")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    # ax2.set_ylabel("Accuracy")
    ax1.legend(loc="upper left", bbox_to_anchor=(1.1, 1))
    # ax2.legend(loc="lower left", bbox_to_anchor=(1.1, 0))

    fig.tight_layout()
    plt.savefig(fp)

    return None


def plot_eg_heat(x: np.ndarray, title: str, fp: Path):
    bins_per_second = 200  # 5ms bins, as in paper
    seconds = x.shape[1] / bins_per_second

    fig, axs = plt.subplots(figsize=(10, 8))

    pcm = axs.pcolormesh(
        np.arange(0, seconds, (1 / bins_per_second)),
        np.arange(0, (x.shape[0])),
        x,
        shading="auto",
        cmap="coolwarm",
        vmin=-0.00001,
        vmax=0.00001,
    )
    axs.set_title(title)
    axs.set_ylabel("Putative Neuron")
    axs.set_xlabel("Time (s)")

    fig.colorbar(pcm, ax=axs, orientation="vertical", label="Voltage (Volts)")

    fig.tight_layout()
    plt.savefig(fp)

    return None


def plot_eg_scatter(x: np.ndarray, title: str, fp: Path):
    bins_per_second = 200  # 5ms bins, as in paper
    n, t = np.where(x)
    t = t / bins_per_second  # type: ignore
    max_t = x.shape[1] / bins_per_second

    fig, axs = plt.subplots(figsize=(10, 8))

    axs.scatter(
        t,
        n,
        s=1,
        alpha=0.5,
        c="#000000",
        marker="s",
    )

    axs.set_title(title)
    axs.set_xlabel("Time (s)")
    axs.set_xlim(0, max_t)
    axs.set_ylabel("Putative Neuron")

    fig.tight_layout()
    plt.savefig(fp)

    return None
