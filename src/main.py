"""
entry point to the pipeline
"""

import argparse

from src.trainer import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrypoint to the LFADS replication pipeline")
    parser.add_argument("--live", action="store_true", help="placeholder", default=False)
    parser.add_argument("--ins", type=str, help="insertion to filter")
    args = parser.parse_args()

    train(**vars(args))
