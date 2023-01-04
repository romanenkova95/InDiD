"""Argument parsers for train.py, test.py and run.py"""
import argparse
import yaml

def get_train_parser():
    """Parse command line arguments for train.py"""

    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument("--model_type", type=str, required=True, help='Model type',
                        choices=["seq2seq", "kl_cpd", "tscp"]) 
                                 # TODO: add baselines
    parser.add_argument("--loss_type", type=str, default=None, help='Loss type for seq2seq model',
                        choices=["indid", "bce", "combined"]) # ??? is it ok?
    parser.add_argument("--experiments_name", type=str, required=True,
                        help='name of dataset',
                        choices=["synthetic_1D", "synthetic_100D", "mnist", "human_activity", "explosion", "road_accidents"])
    parser.add_argument("--seed", type=int, default=102, help="Random seed")
    # parse Boolean argument
    parser.add_argument('--dryrun', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--gpus", type=int, default=0,
                        help="Number of available GPUs.")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of available CPUs.")
    return parser

def get_test_parser():
    """Parse command line arguments for test.py"""

    parser = argparse.ArgumentParser(description='Test your model')
    parser.add_argument("--model_type", type=str, required=True, help='Model type', 
                        choices=["seq2seq", "kl_cpd", "tscp"])
    parser.add_argument("--loss_type", type=str, default=None, help='Loss type for seq2seq model', 
                        choices=["indid", "bce", "combined"])
    parser.add_argument("--experiments_name", type=str, required=True, help='name of dataset', 
                        choices=["synthetic_1D", "synthetic_100D", "mnist", "human_activity", "explosion", "road_accidents"])
    parser.add_argument("--timestamp", type=str, required=True, help='timestamp to be processed')
    parser.add_argument("-tn", "--threshold_number", type=int, default=100, 
                        help='threshold number')
    parser.add_argument("--device", type=str, default="cpu", help='Device for evaluation')

    # boolean
    parser.add_argument("--verbose", default=False, type=lambda x: (str(x).lower() == 'true'), 
                        help='If true, print the metrics to the console.')
    return parser

def get_run_parser():
    """Parse command line arguments for run.py"""

    parser = argparse.ArgumentParser(description='Run experiment')
    parser.add_argument("--model_type", type=str, required=True, help='Model type',
                        choices=["seq2seq", "kl_cpd", "tscp"]) 
                                 # TODO: add baselines
    parser.add_argument("--loss_type", type=str, default=None, help='Loss type for seq2seq model',
                        choices=["indid", "bce", "combined"]) # ??? is it ok?
    parser.add_argument("--experiments_name", type=str, required=True,
                        help='name of sdataset',
                        choices=["synthetic_1D", "synthetic_100D", "mnist", "human_activity", "explosion", "road_accidents"])
    parser.add_argument("--seed", type=int, default=102, help="Random seed")
    parser.add_argument("--gpus", type=int, default=0,
                        help="Number of available GPUs.")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of available CPUs.")
    parser.add_argument("-tn", "--threshold_number", type=int, default=100, 
                        help='threshold number')
    # boolean
    parser.add_argument("--verbose", default=False, type=lambda x: (str(x).lower() == 'true'), 
                        help='If true, print the metrics to the console.')
    return parser

def get_args(parser: argparse.ArgumentParser) -> dict:
    """Parse command line arguments and parameteres defined in .yaml config files"""
    # get command line arguments
    args = parser.parse_args()
    args = dict(vars(args))

    if args["experiments_name"] in ["explosion", "road_accidents"]:
       experiments_name = "video"
    else: 
        experiments_name = args["experiments_name"]
    
    # e.g. of a correct config path: "configs/synthetic_1D_seq2seq.yaml", 
    # "configs/video_seq2seq.yaml" -- for "explosion" or "road accidents"
    path_to_config = "configs/" + experiments_name + "_" + args["model_type"] + ".yaml"

    # read arguments from a default config file
    with open(path_to_config, 'r') as f:
        args_config = yaml.safe_load(f.read())

    args.update(args_config)
    return args