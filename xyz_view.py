from psi_tools import *

from lcao_tools import *

import parse_xyz

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("inputfile")
parser.add_argument("--orbitals", type=str, choices=["all", "valence", "pz"], required=True)
parser.add_argument("--plot", type=str, choices=["all", "frontier"], required=True)
args = parser.parse_args()
print(f"Parsing file {args.inputfile}")

atom_positions = []
atom_charges = []

with open(args.inputfile, "r") as file:
	atom_positions, atom_charges = parse_xyz.parse(file.read())

valence_charges = parse_xyz.valence_charges(atom_charges)
print("Atom positions (m):", atom_positions)
print("Atom charges:", atom_charges)
print("Valence charges:", valence_charges)

molecule_extent = np.max(np.abs(atom_positions))
print("Molecule max extent (m):", molecule_extent)
molecule_extent_buffered = molecule_extent * 1.2 + 1 * bohr_radius
print("Buffered plot radius (m):", molecule_extent_buffered)

lattice_shape = (32, 32, 32)
latticeConfig(lattice_shape, molecule_extent_buffered)

print("Adding nudge", 0.5*latticeLengthPerStep())
for i in range(len(atom_positions)):
	atom_positions[i] += np.array([0.5, 0.5, 0.5])*latticeLengthPerStep()

orbital_funcs = []
if args.orbitals == "all":
	orbital_funcs = parse_xyz.default_orbital_functions(atom_charges)
elif args.orbitals == "valence":
	orbital_funcs = parse_xyz.valence_orbital_functions(atom_charges)
elif args.orbitals == "pz":
	orbital_funcs = parse_xyz.z_orbital_functions(atom_charges)
atomic_orbitals = []
total_potential = np.zeros(lattice_shape, dtype = complex)
generate_orbitals(atom_positions, atom_charges, orbital_funcs, atomic_orbitals, total_potential)
print(f"Generated {len(atomic_orbitals)} orbitals")

eigs = calculateMOcoeffs(atomic_orbitals, total_potential)
print("Calculated MO coeffs")

MOs = getMOs(atomic_orbitals, eigs)

print("Eigenvalues (eV):", np.sort(eigs.eigenvalues) / charge_e)

energies = np.real(np.sort(eigs.eigenvalues))
#totalEnergy = groundStateEnergy(energies, atom_positions, atom_charges)
#print(f"Ground State Energy: {totalEnergy / charge_e} eV")
homo_lumo_gap_energy = None
if args.orbitals == "all":
	homo_lumo_gap_energy = homo_lumo_gap(energies, atom_charges)
elif args.orbitals == "valence":
	homo_lumo_gap_energy = homo_lumo_gap(energies, valence_charges)
elif args.orbitals == "pz":
	homo_lumo_gap_energy = homo_lumo_gap(energies, pz_charges)

print(f"HOMO-LUMO Gap: {homo_lumo_gap_energy / charge_e} eV")

if args.plot == "all":
	if args.orbitals == "valence":
		plotMOs(atomic_orbitals, eigs, atom_positions, valence_charges, quantile=0.5, num_cols=4)
	else:
		plotMOs(atomic_orbitals, eigs, atom_positions, atom_charges, quantile=0.5, num_cols=4) # todo what about pz
elif args.plot == "frontier":
	#plotHOMOLUMO(atomic_orbitals, eigs, valence_charges, quantile=0.5)
	if args.orbitals == "valence":
		plotHOMOLUMO_more(atomic_orbitals, eigs, atom_positions, valence_charges, quantile=0.5)
	else:
		plotHOMOLUMO_more(atomic_orbitals, eigs, atom_positions, atom_charges, quantile=0.5)