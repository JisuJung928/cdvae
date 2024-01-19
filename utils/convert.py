import os
import sys
import torch
import argparse

from ase import Atoms, io


def main(args):
    data = torch.load(args.input_file)
    if os.path.exists(args.output_path):
        sys.exit(f"{args.output_path} already exists.\n")
    os.mkdir(args.output_path)

    num_evals = data["atom_types"].shape[0]
    for i in range(num_evals):
        frac_coords = data["frac_coords"][i]
        atom_types = data["atom_types"][i]
        lengths = data["lengths"][i]
        angles = data["angles"][i]
        num_atoms = data["num_atoms"][i]
        frac_coords = torch.split(frac_coords, num_atoms.tolist())
        atom_types = torch.split(atom_types, num_atoms.tolist())
        if args.traj:
            all_frac_coords = torch.split(
                data["all_frac_coords_stack"][i], num_atoms.tolist(), dim=1
            )
            all_atom_types = torch.split(
                data["all_atom_types_stack"][i], num_atoms.tolist(), dim=1
            )
        for j, (frac_coord, atom_type, length, angle) in enumerate(
            zip(frac_coords, atom_types, lengths, angles)
        ):
            atoms = Atoms(
                scaled_positions=frac_coord.tolist(),
                numbers=atom_type.tolist(),
                cell=length.tolist() + angle.tolist(),
            )
            io.write(f"{args.output_path}/POSCAR_{i}_{j}", atoms, vasp5=True, sort=True)
            if args.traj:
                for k, (frac_coord, atom_type) in enumerate(
                    zip(all_frac_coords[j], all_atom_types[j])
                ):
                    atoms = Atoms(
                        scaled_positions=frac_coord.tolist(),
                        numbers=atom_type.tolist(),
                        cell=length.tolist() + angle.tolist(),
                    )
                    io.write(
                        f"{args.output_path}/POSCAR_{i}_{j}_{k}",
                        atoms,
                        vasp5=True,
                        sort=True,
                    )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("input_file", help="path of lammps structure files", type=str)
    args.add_argument("output_path", help="path of lammps log files", type=str)
    args.add_argument("--traj", help="extract trajectory", action="store_true")

    args = args.parse_args()
    main(args)
