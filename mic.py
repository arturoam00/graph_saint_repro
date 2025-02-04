#!/usr/bin/env python3

import argparse
import os
import re
import statistics


def main():
    parser = argparse.ArgumentParser(
        description="Extract mic values from files and compute their mean and standard deviation."
    )
    parser.add_argument(
        "directory",
        type=str,
        default="./log_files",
        help="Directory path to search for files",
    )
    parser.add_argument("prefix", type=str, help="Prefix of the files to search for")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: '{args.directory}' is not a valid directory.")
        exit(1)

    mic_values = []

    pattern = re.compile(
        r"INFO:src.train:TEST:\s*loss\s*=\s*(\d+(?:\.\d+)?)\s+mic\s*=\s*(\d+(?:\.\d+)?)\s+mac\s*=\s*(\d+(?:\.\d+)?)"
    )

    for filename in os.listdir(args.directory):
        if filename.startswith(args.prefix):
            file_path = os.path.join(args.directory, filename)
            if os.path.isfile(file_path):
                try:
                    with open(file_path, "r") as file:
                        content = file.read()
                        match = pattern.search(content)
                        if match:
                            mic_value = float(match.group(2))
                            mic_values.append(mic_value)
                        else:
                            print(f"Pattern not found in file: {filename}")
                except Exception as e:
                    print(f"Error reading file '{filename}': {e}")

    if not mic_values:
        print("No valid 'mic' values were found in the files.")
        return

    mean_val = statistics.mean(mic_values)
    stdev_val = statistics.stdev(mic_values) if len(mic_values) > 1 else 0.0

    print("Mean of mic values: {:.3f}".format(mean_val))
    print("Standard deviation of mic values: {:.3f}".format(stdev_val))


if __name__ == "__main__":
    main()
