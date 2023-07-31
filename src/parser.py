import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="set optional parameters")

    parser.add_argument(
        "--nfeatures",
        type=int,
        default=None,
        help="Adjust number of features to detect [default: %(default)s ]",
    )
    parser.add_argument(
        "--n_keypoints_trajectories",
        type=int,
        default=40,
        help="Number frames to keep the features tracked [default: %(default)s ]",
    )
    parser.add_argument(
        "--detection_interval",
        type=int,
        default=5,
        help="Interval of finding new features [default: %(default)s ]",
    )
    return parser.parse_args()
