import argparse
import os
import shutil

from data_handler import CVLDataGenerator, Icdar2013DataGenerator, FiremakerDataGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="The dataset to split, can be CVL, Firemaker, ICDAR2013",
                        default="CVL")
    parser.add_argument("-n", "--splits", type=int,
                        help="The number of subsplits to split the train and validation split into for X-fold-cross-validation")
    config = vars(parser.parse_args())

    splits = config["splits"]
    if splits < 2:
        raise RuntimeError("Splits must be > 1")

    target_folder = os.path.join(os.path.dirname(__file__), "result")

    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)

    os.makedirs(target_folder, exist_ok=True)
    os.makedirs(os.path.join(target_folder, "test"))
    os.makedirs(os.path.join(target_folder, "train_val"))

    dataset_type = config["dataset"]
    print("Selected Dataset: ", dataset_type)
    if dataset_type == "CVL":
        CVLDataGenerator(splits)
    elif dataset_type == "Firemaker":
        FiremakerDataGenerator(splits)
    elif dataset_type == "ICDAR2013":
        Icdar2013DataGenerator(splits)
    else:
        raise RuntimeError("Unknown dataset")


if __name__ == '__main__':
    main()
