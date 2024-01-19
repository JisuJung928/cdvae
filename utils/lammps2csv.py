import os
import sys
import pandas as pd
import argparse
from braceexpand import braceexpand

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
    python lammps2csv.py "structure_path" "log_path" --atomic_symbols Si O ...
    Be careful for quotation marks.
    """
    structure_list = list(braceexpand(args.structure_path))
    log_list = list(braceexpand(args.log_path))
    if len(structure_list) != len(log_list):
        sys.exit("Mismatch!\n")
    csv = pd.DataFrame(columns=["material_id", "free_energy", "cif"])
    for count, (structure, log) in enumerate(zip(structure_list, log_list)):
        try:
            structure = io.read(structure, format="lammps-data", style="atomic")
        except Exception:
            print(f"{structure} does not exist\n")
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
    args.add_argument("structure_path", help="path of lammps structure files", type=str)
    args.add_argument("log_path", help="path of lammps log files", type=str)
    args.add_argument(
        "--atomic_symbols", nargs="*", help="Atomic symbols", required=True
    )

    args = args.parse_args()
    main(args)
