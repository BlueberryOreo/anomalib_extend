from argparse import ArgumentParser, Namespace
import logging
import warnings


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="patchcore", help="Name of the algorithm to train/test")
    parser.add_argument("--config", type=str, default=False, help="Path to a model config file")
    parser.add_argument("--log-level", type=str, default="INFO", help="<DEBUG, INFO, WARNING, ERROR>")

    args = parser.parse_args()
    return args


def train():
    args = get_args()
    # print(args)
    if args.log_level == "ERROR":
        warnings.filterwarnings("ignore")

    pass


if __name__ == '__main__':
    train()
