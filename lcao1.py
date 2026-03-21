from psi_tools import *

from lcao_tools import *

#############
# Setup
#############

lattice_shape = (32, 32, 32)
latticeConfig(lattice_shape, 3.0 * bohr_radius)


def run(iteration, params, output_data_table):
	atom_positions = []
	atom_charges = []
	atomic_orbitals = []
	orbital_funcs = []
	total_potential = np.zeros(lattice_shape, dtype = complex)

	nudge = np.array([0.5, 0.5, 0.5])*latticeLengthPerStep()
	atom_positions.append(np.array([-0.5*params["spacing"],0,0])+nudge)
	atom_charges.append(2)
	orbital_funcs.append([orbital_1s])
	#orbital_funcs.append([orbital_1s, orbital_2s, orbital_2p_z, orbital_2p_left, orbital_2p_right])
	atom_positions.append(np.array([0.5*params["spacing"],0,0])+nudge)
	atom_charges.append(2)
	orbital_funcs.append([orbital_1s])
	#orbital_funcs.append([orbital_1s, orbital_2s, orbital_2p_z, orbital_2p_left, orbital_2p_right])
	print("Setup atom positions")

	generate_orbitals(atom_positions, atom_charges, orbital_funcs, atomic_orbitals, total_potential)
	print("Generated orbitals")

	eigs = calculateMOcoeffs(atomic_orbitals, total_potential)
	print("Calculated MO coeffs")

	MOs = getMOs(atomic_orbitals, eigs)

	print("Eigenvalues:", np.sort(eigs.eigenvalues) / charge_e)

	energies = np.real(np.sort(eigs.eigenvalues))
	totalEnergy = groundStateEnergy(energies, atom_positions, atom_charges)
	print(f"Ground State Energy: {totalEnergy / charge_e} eV")
	output_data_table[iteration][0] = totalEnergy


	
	#plotMOs(atomic_orbitals, eigs, quantile=0.5, num_cols=5)
	#exit()



#############
# Iteration
#############

iterations = 17

output_data_table = np.zeros((iterations, 1))
labels = ["Bonding MO Energy", "Antibonding MO Energy", "Nuclear Repulsion", "Electron Repulsion (n/a)", "Total Energy"]

spacing_array = np.linspace(0.5*bohr_radius, 3.0*bohr_radius, iterations)

for i in range(iterations):
	print(f"Iteration {i}, bond length {spacing_array[i] / bohr_radius} bohr")
	params = {
		"spacing": spacing_array[i]
	}
	run(i, params, output_data_table)

print(output_data_table)
print(f"Lowest Total Energy: {np.min(output_data_table[:, 0]) / charge_e} eV at index {np.argmin(output_data_table[:, 0])}, spacing {spacing_array[np.argmin(output_data_table[:, 0])] / bohr_radius} bohr radii")

hartree_per_eV = 1 / 27.211386
plt.plot(spacing_array / bohr_radius, output_data_table[:, 0] / charge_e * hartree_per_eV)
#plt.legend(labels)
plt.ylabel("Total Energy (Hartree)")
plt.xlabel("Bond Length (Bohr Radii)")
plt.title("He2 Ground State Energy vs Nuclei Separation, via LCAO of 1s, No e- interaction")
plt.show()

