import os
import numpy as np
import pandas as pd
import argparse

from ase import io
from ase.data import atomic_numbers
from ase.build.tools import sort


def parse_log(filename):
    """
    parse_log reads the last energy as written in default thermo_style.
    """
    with open(filename, "r") as fp:
        data = list(reversed(fp.readlines()))
        count = 0
        for line in data:
            count += 1
            if "Loop time" in line:
                break
        line = data[count]
        # E_pair
        energy = float(line.split()[2])
    return energy


def main(args):
    """
    python lammps2csv.py --structure_files structure_1 structure_2 ...
                         --log_files log_1 log_2 ...
                         --atomic_symbols Si O ...
    """
    assert len(args.structure_files) == len(
        args.log_files
    ), "# of structure and log are different."
    csv = pd.DataFrame(columns=["material_id", "free_energy", "cif"])
    for count, (structure, log) in enumerate(zip(args.structure_files, args.log_files)):
        try:
            structure = io.read(structure, format="lammps-data", style="atomic")
        except Exception:
            print(f"{structure} does not exist\n")
            continue
        distance = structure.get_all_distances(mic=True)
        distance = distance[np.triu_indices(len(structure), 1)]
        if np.min(distance) < 0.5:
            print(f"{structure} has the site occupancy larger than 1.\n")
            continue
        for i in range(len(structure)):
            structure.numbers[i] = atomic_numbers[
                args.atomic_symbols[structure.numbers[i] - 1]
            ]
        structure = sort(structure)
        energy = parse_log(log)
        io.write("tmp.cif", structure, format="cif")
        with open("tmp.cif", "r") as fp:
            cif = fp.read()
        csv = pd.concat(
            [
                csv,
                pd.DataFrame(
                    data=[[count, energy, cif]],
                    columns=["material_id", "free_energy", "cif"],
                ),
            ],
            ignore_index=True,
        )
        os.remove("tmp.cif")
        print(f"{count + 1} / {len(args.structure_files)}")
    # split data into train(6):valid(2):test(2)
    train = csv.sample(frac=0.6)
    test = csv.drop(train.index)
    valid = test.sample(frac=0.5)
    test = test.drop(valid.index)
    # save it
    train.to_csv("train.csv")
    valid.to_csv("val.csv")
    test.to_csv("test.csv")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--structure_files", nargs="*", help="lammps structure files", required=True
    )
    args.add_argument("--log_files", nargs="*", help="lammps log files", required=True)
    args.add_argument(
        "--atomic_symbols", nargs="*", help="Atomic symbols", required=True
    )

    args = args.parse_args()
    main(args)
