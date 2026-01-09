from pathlib import Path

import pandas as pd  # type: ignore
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore

from src.data import OneDs  # type: ignore
from src.model import LFADS
from src.utils import get_available_device, plot_eg_heat, plot_eg_scatter, plot_metrics
from src.utils import wd


@torch.no_grad()
def update_checkpoint(
    step: int,
    device: torch.device,
    model: LFADS,
    loader: DataLoader,
    working_dir: Path,
    train_steps: list[int],
    test_steps: list[int],
    train_losses: list[float],
    test_loss: list[float],
):
    # perform tests
    losses = []
    for i, x in tqdm(enumerate(loader), desc="Testing Loop", total=25, leave=False):
        if i >= 25:
            break
        x = x.to(device)
        rates, factors, states, inferred_inputs, loss = model(x)
        losses.append(loss.detach().cpu().float())

    # save an example
    x = x[0].detach().cpu().numpy()  # type: ignore
    rates = rates[0, ...].detach().cpu().numpy()  # type: ignore

    # save test metrics for plot
    test_steps.append(step)
    test_loss.append(float(sum(losses) / len(losses)))

    # save the state dictionary
    torch.save(model.state_dict(), (working_dir / "results" / f"checkpoint_{step}.pth"))

    # gather the metrics
    train_metrics_df = pd.DataFrame(
        {
            "step": train_steps,
            "train_loss": train_losses,
        }
    )
    test_metrics_df = pd.DataFrame(
        {
            "step": test_steps,
            "test_loss": test_loss,
        }
    )
    df = pd.merge(train_metrics_df, test_metrics_df, on="step", how="outer")

    # save and plot the metrics
    df.sort_values("step", ascending=False, inplace=True)
    df.to_csv(working_dir / "results" / "metrics.csv")
    plot_metrics((train_metrics_df, test_metrics_df), fp=(working_dir / "results" / "metrics.png"))

    # plot an example
    plot_eg_scatter(
        x,
        title="Putative Neural Spikes (Measured)",
        fp=(working_dir / "results" / f"example_measured_spikes_{step}.png"),
    )
    plot_eg_heat(
        rates,
        title="Predicted Neural Spiking Rates",
        fp=(working_dir / "results" / f"example_predicted_rates_{step}.png"),
    )

    return None


def train(
    steps: int = int(800),
    checkpoint_freq: int = 100,
    lr: float = 0.0001,
    betas: tuple = (0.9, 0.999),  # from paper
    eps: float = 0.1,  # from paper
    ins: str = "",  # if provided, only consider this one probe/recording session
    **kwargs,
) -> None:
    """
    Training loop for NeuroGPT
    """
    # objects
    device = get_available_device()
    model: LFADS = LFADS()
    train_set = OneDs(True, ins)
    test_set = OneDs(False, ins)
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=4, shuffle=True, num_workers=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps)
    working_dir = wd()

    train_steps: list[int] = []
    test_steps: list[int] = []
    train_losses: list[float] = []
    test_loss: list[float] = []

    # setup
    data_iter = iter(train_loader)
    model.to(device)

    # training_loop
    for i in tqdm(range(steps), desc="Training Loop", total=steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        x = batch

        x = x.to(device)

        model.zero_grad()
        _, _, _, _, loss = model(x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=200.0)  # from paper
        optimizer.step()

        train_steps.append(i)
        train_losses.append(float(loss.detach().cpu()))

        # handle checkpoints
        if i % checkpoint_freq == 0:
            model.eval()
            update_checkpoint(
                i,
                device,
                model,
                test_loader,
                working_dir,
                train_steps,
                test_steps,
                train_losses,
                test_loss,
            )
            model.train()

    model.eval()
    update_checkpoint(
        (steps - 1),
        device,
        model,
        test_loader,
        working_dir,
        train_steps,
        test_steps,
        train_losses,
        test_loss,
    )
    return None
