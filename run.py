"""Script that runs a full experiment: initialization + training + evaluation."""
from collections import defaultdict
from itertools import product
from typing import Dict
import yaml
from pathlib import Path
from subprocess import PIPE, Popen
import pickle
import json
from termcolor import cprint

from utils.arg_parsing import get_run_parser, get_args

def main(args: Dict) -> None:
    train_cmd = ["python3", "train.py", "--model_type", args["model_type"]]

    if args["loss_type"] is not None:
       train_cmd.extend(["--loss_type", args["loss_type"]])
    
    train_cmd.extend([
        "--experiments_name", args["experiments_name"],
        "--seed", str(args["seed"]),
        "--dryrun", "False", # Currently, run-all is impossible without saving the results
        "--num_workers", str(args["num_workers"]),
        "--gpus", str(args["gpus"])
        ])

    cprint("Train command:\n" + " ".join(train_cmd), "red")

    with Popen(train_cmd, stdout=PIPE, bufsize=1,
               universal_newlines=True) as proc:

        line_last = ""
        assert proc.stdout is not None, f'proc.stdout is None'
        for line in proc.stdout:
            print(line, end="")
            line_last = line

        timestamp = line_last.rstrip()
    
    test_cmd = ["python3", "test.py", "--model_type", args["model_type"]]

    if args["loss_type"] is not None:
       test_cmd.extend(["--loss_type", args["loss_type"]])

    device = "cuda" if args["gpus"] > 0 else "cpu"

    test_cmd.extend([
        "--experiments_name", str(args["experiments_name"]),
        "--timestamp", timestamp,
        "--threshold_number", str(args["threshold_number"]),
        "--device", device,
        "--verbose", str(args["verbose"])
        ]) 

    cprint("Test command:\n" + " ".join(test_cmd), "red")

    with Popen(test_cmd, stdout=PIPE, bufsize=1, universal_newlines=True) as proc:
        assert proc.stdout is not None, f'proc.stdout is None'
        for line in proc.stdout:
            print(line, end="")

if __name__ == "__main__":
    parser = get_run_parser()
    args = get_args(parser)
    main(args)