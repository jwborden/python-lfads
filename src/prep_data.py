"""
Run this script to collect the IBL's data into .parquet files at ./data/

It wil only gather data corresponding to neurons in
these motor cortex regions:
id,      name,                            acronym
500,     Somatomotor areas,               MO
107,     Somatomotor areas Layer 1,       MO1
219,     Somatomotor areas Layer 2/3,     MO2/3
299,     Somatomotor areas Layer 5,       MO5
644,     Somatomotor areas Layer 6a,      MO6a
947,     Somatomotor areas Layer 6b,      MO6b
985,     Primary motor area,              MOp
320,     Primary motor area Layer 1,      MOp1
943,     Primary motor area Layer 2/3,    MOp2/3
648,     Primary motor area Layer 5,      MOp5
844,     Primary motor area Layer 6a,     MOp6a
882,     Primary motor area Layer 6b,     MOp6b
993,     Secondary motor area,            MOs
656,     Secondary motor area layer 1,    MOs1
962,     Secondary motor area layer 2/3,  MOs2/3
767,     Secondary motor area layer 5,    MOs5
1021,    Secondary motor area layer 6a,   MOs6a
1085,    Secondary motor area layer 6b,   MOs6b
"""

import json
import os
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore

from brainbox.io.one import SpikeSortingLoader  # type: ignore
from iblutil.util import Bunch  # type: ignore
from one.alf.io import AlfBunch  # type: ignore
from one.api import OneAlyx, ONE  # type: ignore # Docs: https://int-brain-lab.github.io/ONE/

from src.utils import log, wd


ACRONYMS = [
    "MO",
    "MO1",
    "MO2/3",
    "MO5",
    "MO6a",
    "MO6b",
    "MOp",
    "MOp1",
    "MOp2/3",
    "MOp5",
    "MOp6a",
    "MOp6b",
    "MOs",
    "MOs1",
    "MOs2/3",
    "MOs5",
    "MOs6a",
    "MOs6b",
]


def folder_size_gb(path: str) -> float:
    """
    Compute the total size of all files in a folder (recursively) in gigabytes.

    :param path: path to the folder

    :return: total size in gigabytes.
    """
    total_bytes = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            try:
                fp = os.path.join(root, f)
                total_bytes += os.path.getsize(fp)
            except OSError:
                # skip files that cannot be accessed
                continue
    return total_bytes / 1e9  # convert bytes to GB


def get_json_contents(fp) -> dict:
    with open(fp, "r") as f:
        return json.load(f)


def set_json_contents(fp, body) -> None:
    with open(fp, "w") as f:
        json.dump(body, f, indent=2)
    return None


def save_insertions(o: OneAlyx, fp) -> list[str]:
    """
    The IBL data is assigned locations in an atlas:

    :param o: OneAlyx object for interfacing with the ONE api
    :param fp: filepath for the json file to be saved

    :return: the list of insertions
    """
    tag = "2024_Q2_IBL_et_al_BWM_iblsort"  # tag for IBL's most recent data release
    insertions: list[str] | set = set()
    for (
        ac
    ) in ACRONYMS:  # gather from insertions with these acronyms, but only insertions with these
        ins: list[str] = o.search_insertions(  # list of insertions
            atlas_acronym=ac, tag=tag, query_type="remote"
        )  # type: ignore
        insertions = insertions.union(set(ins))  # type: ignore
        assert len(insertions) >= len(ins)
    insertions = [str(insertion) for insertion in insertions]
    set_json_contents(fp, insertions)
    return insertions


def process(pid: str, one_alyx: OneAlyx, d: Path) -> None:
    """
    Get spike data for one insertion during a recording session and save it to parquet files

    :param pid: pid
    :param one_alyx: one_alyx
    :param d: data path
    """
    bins_per_s = 200  # 5ms bins, as in Pandarinath et al.
    train_dir = d / "train"  # the directory to hold .parquet files with training data
    test_dir = d / "test"  # the directory to hold .parquet files with test data

    # get IBL data
    spike_loader = SpikeSortingLoader(pid=pid, one=one_alyx)
    spike_sorting_data: tuple[AlfBunch, AlfBunch, Bunch] = spike_loader.load_spike_sorting()  # type: ignore
    spikes, *_ = spike_sorting_data
    merged_clusters: AlfBunch = spike_loader.merge_clusters(*spike_sorting_data)  # type: ignore
    merged_clusters_df = merged_clusters.to_df()
    spikes_df = spikes.to_df()

    # filter to only include motor neurons
    merged_clusters_df = merged_clusters_df[merged_clusters_df["acronym"].isin(ACRONYMS)]

    # wrangle into one dataframe
    cluster_channel_map = (
        merged_clusters_df[["cluster_id", "channels"]]
        .copy()
        .sort_values(by="channels", ascending=True)
        .reset_index(drop=True)
        .reset_index(drop=False)
        .rename(inplace=False, columns={"index": "cluster_channel_id"})  # sort by channels
    )
    spikes_df = cluster_channel_map.merge(
        spikes_df, left_on="cluster_id", right_on="clusters", how="left"
    )[["cluster_channel_id", "times"]]
    spikes_df["time_bin"] = (np.floor(spikes_df["times"] * bins_per_s)).astype(int)

    # make an n-by-t spike-train matrix
    n = cluster_channel_map.shape[0]  # putative neurons: spike clusters sorted by channel
    t = spikes_df["time_bin"].max()  # 5 ms time bins
    spikes_matrix = np.zeros((n, t + 1))  # type: ignore
    spikes_matrix[
        spikes_df["cluster_channel_id"].values,  # type: ignore
        spikes_df["time_bin"].values,  # type: ignore
    ] = int(1)  # 1 represents a spike
    spikes_matrix = spikes_matrix.astype(np.bool)

    # save one parquet file per <time_window> bins
    time_window = 160  # 160 bins is 800ms, the same as Pandarinath et al.
    neuron_window = 128
    train_split = 0.8

    if spikes_matrix.shape[0] < neuron_window or spikes_matrix.shape[1] < time_window:
        return None

    for i in range(0, (spikes_matrix.shape[1] - time_window), time_window):
        j = i + time_window
        max_k = spikes_matrix.shape[0] - neuron_window - 1
        k = random.randint(0, max_k)
        l = k + neuron_window  # noqa: E741
        x = spikes_matrix[k:l, i:j]
        assert x.shape == (neuron_window, time_window)
        target_dir = train_dir if random.random() < train_split else test_dir
        f_name = f"{pid}_{i}-{j}_{k}-{l}.parquet"
        fp = target_dir / f_name
        pd.DataFrame(x).to_parquet(fp)
        # the file could then be read as:
        # pd.read_parquet(fp).values

    return None


if __name__ == "__main__":
    start_time = datetime.now(timezone.utc)

    max_hours = 10  # end safely after about 10 hours
    max_gb = 224  # end safely after about 224 GB

    working_dir = wd()
    data_dir = working_dir / "data"
    cache_dir = working_dir / "cache" / "one"
    insertions_list_fp = data_dir / "insertions.json"

    one: OneAlyx = ONE(
        cache_dir=cache_dir,
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",  # public-access password
        silent=True,  # don't print progress, etc.
    )  # type: ignore

    insertions: list[str] = (
        get_json_contents(insertions_list_fp)
        if insertions_list_fp.exists()
        else save_insertions(one, insertions_list_fp)  # type: ignore
    )
    random.shuffle(insertions)

    train_dir = data_dir / "train"  # the directory to hold .parquet files with training data
    test_dir = data_dir / "test"  # the directory to hold .parquet files with test data
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for insertion in insertions:
        now = datetime.now(timezone.utc)
        hours_elapsed = (now - start_time).total_seconds() / 3600
        if hours_elapsed > max_hours:
            break

        if cache_dir.exists():
            shutil.rmtree(cache_dir)  # clean up the cache
        sized = folder_size_gb(str(data_dir))
        if sized > max_gb:  # check the size of downloads
            break

        log(
            f"processing insertion {insertion} at {now.strftime('%d/%m/%Y, %H:%M:%S')}, {np.round(hours_elapsed, 2)} hours and {sized}GB into processing."
        )
        # insertion = "8c732bf2-639d-496c-bf82-464bc9c2d54b" # dev only
        try:
            process(insertion, one, data_dir)
        except Exception as e:
            log(str(e), "ERROR")

    if cache_dir.exists():
        shutil.rmtree(cache_dir)  # clean up the cache
    log("done")
