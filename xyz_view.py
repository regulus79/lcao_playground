from psi_tools import *

from lcao_tools import *

import parse_xyz

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("inputfile")
parser.add_argument("--orbitals", type=str, choices=["all", "valence", "pz"], required=True)
parser.add_argument("--plot", type=str, choices=["all", "frontier"], required=True)
parser.add_argument("--shielding", type=str, choices=["none", "valence", "table"], required=True)
parser.add_argument("--quantile", type=float, default=0.5)
parser.add_argument("--buffer", type=float, default=3)
args = parser.parse_args()
print(f"Parsing file {args.inputfile}")

atom_positions = []
atom_charges = []

with open(args.inputfile, "r") as file:
	atom_positions, atom_charges = parse_xyz.parse(file.read())

effective_charges = None
print("Atom positions (m):", atom_positions)
print("Atom charges:", atom_charges)
if args.shielding == "none":
	effective_charges = atom_charges
elif args.shielding == "valence":
	effective_charges = parse_xyz.valence_charges(atom_charges)
elif args.shielding == "table":
	effective_charges = parse_xyz.table_effective_charges(atom_charges)
print("Effective charges:", effective_charges)

molecule_extent = np.max(np.abs(atom_positions))
print("Molecule max extent (m):", molecule_extent)
molecule_extent_buffered = molecule_extent + args.buffer * bohr_radius
print("Buffered plot radius (m):", molecule_extent_buffered)

lattice_shape = (32, 32, 32)
latticeConfig(lattice_shape, molecule_extent_buffered)

print("Adding nudge", 0.5*latticeLengthPerStep())
for i in range(len(atom_positions)):
	atom_positions[i] += np.array([0.5, 0.5, 0.5])*latticeLengthPerStep()

orbital_funcs = []
atomic_orbitals = []
num_electrons = None
total_potential = np.zeros(lattice_shape, dtype = complex)
if args.orbitals == "all":
	orbital_funcs = parse_xyz.default_orbital_functions(atom_charges)
	generate_orbitals(atom_positions, effective_charges, orbital_funcs, atomic_orbitals, total_potential)
	num_electrons = sum(atom_charges)
elif args.orbitals == "valence":
	orbital_funcs = parse_xyz.valence_orbital_functions(atom_charges)
	generate_orbitals(atom_positions, effective_charges, orbital_funcs, atomic_orbitals, total_potential)
	num_electrons = sum(parse_xyz.valence_charges(atom_charges))
elif args.orbitals == "pz":
	orbital_funcs = parse_xyz.z_orbital_functions(atom_charges)
	generate_orbitals(atom_positions, effective_charges, orbital_funcs, atomic_orbitals, total_potential) # TODO
	num_electrons = sum(atom_charges)
print(f"Generated {len(atomic_orbitals)} orbitals")

eigs = calculateMOcoeffs(atomic_orbitals, total_potential)
print("Calculated MO coeffs")

MOs = getMOs(atomic_orbitals, eigs)

print("Eigenvalues (eV):", np.sort(eigs.eigenvalues) / charge_e)
print(eigs)
print("Eigenvectors:")
for i in np.argsort(eigs.eigenvalues):
	print(np.round(eigs.eigenvectors[:, i], 4))

energies = np.real(np.sort(eigs.eigenvalues))
#totalEnergy = groundStateEnergy(energies, atom_positions, atom_charges)
#print(f"Ground State Energy: {totalEnergy / charge_e} eV")
homo_lumo_gap_energy = homo_lumo_gap(energies, num_electrons)

print(f"HOMO-LUMO Gap: {homo_lumo_gap_energy / charge_e} eV")

if args.plot == "all":
	plotMOs(atomic_orbitals, eigs, atom_positions, num_electrons, quantile=args.quantile, num_cols=4)
elif args.plot == "frontier":
	plotHOMOLUMO_more(atomic_orbitals, eigs, atom_positions, num_electrons, quantile=args.quantile)