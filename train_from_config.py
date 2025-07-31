import argparse
import yaml
import sys

import train


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run train.py using parameters from a YAML config file.")
    parser.add_argument(
        "config", type=str, help="Path to YAML configuration file")
    args, unknown = parser.parse_known_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # Build argument list for train.py
    argv = ["train.py"]
    for key, value in config.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        else:
            argv.extend([flag, str(value)])
    argv.extend(unknown)

    sys.argv = argv
    train.main()


if __name__ == "__main__":
    main()
