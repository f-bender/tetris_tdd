# ruff: noqa: T201

import ast
import sys
from pathlib import Path
from pprint import pformat

import pandas as pd


def main(argv: list[str]) -> None:
    n = int(argv[0]) if argv else 2

    report = pd.read_csv(Path(__file__).parent / "report.csv")

    for key in ("checkpoint", "top_k"):
        report.loc[:, key] = report["extra_info"].apply(lambda dict_repr, key=key: ast.literal_eval(dict_repr)[key])

    k1 = report[report["top_k"] == 1]
    kn = report[report["top_k"] == n]

    merged = k1.merge(kn, on="checkpoint", how="inner")

    print(f"Improvements of Top 1 over Top {n}")
    print()
    print()
    print("Raw diffs:")
    print()
    print(f"Mean Score:\n{pformat([round(x, 2) for x in merged['mean_score_x'] - merged['mean_score_y']])}")
    print(f"Median Score:\n{pformat([round(x, 2) for x in merged['median_score_x'] - merged['median_score_y']])}")
    print(f"Max Score:\n{pformat([round(x, 2) for x in merged['max_score_x'] - merged['max_score_y']])}")
    print(f"Min Score:\n{pformat([round(x, 2) for x in merged['min_score_x'] - merged['min_score_y']])}")
    print()
    print()
    print("Stats:")
    print()
    print(f"Mean Score:\n{(merged['mean_score_x'] - merged['mean_score_y']).describe()}")
    print(f"Median Score:\n{(merged['median_score_x'] - merged['median_score_y']).describe()}")
    print(f"Max Score:\n{(merged['max_score_x'] - merged['max_score_y']).describe()}")
    print(f"Min Score:\n{(merged['min_score_x'] - merged['min_score_y']).describe()}")


if __name__ == "__main__":
    main(sys.argv[1:])
