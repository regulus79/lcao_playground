from psi_tools import *

from lcao_tools import *

#############
# Setup
#############

lattice_shape = (40, 40, 40)
latticeConfig(lattice_shape, 3.0 * bohr_radius)


def run(iteration, params, output_data_table):
	atom_positions = []
	atom_charges = []
	atomic_orbitals = []
	orbital_funcs = []
	total_potential = np.zeros(lattice_shape, dtype = complex)

	nudge = np.array([0.5, 0.5, 0.5])*latticeLengthPerStep()
	spacing = 1.81*0.4

	atom_positions.append(np.array([spacing,spacing/2,0])*bohr_radius+nudge)
	atom_charges.append(1)
	orbital_funcs.append([orbital_1s])
	
	atom_positions.append(np.array([0,-spacing/2,0])*bohr_radius+nudge)
	atom_charges.append(8)
	orbital_funcs.append([orbital_1s, orbital_2p_z, orbital_2p_left, orbital_2p_right])
	#orbital_funcs.append([orbital_1s, orbital_2s, orbital_2p_z, orbital_2p_left, orbital_2p_right])
	
	atom_positions.append(np.array([-spacing,spacing/2,0])*bohr_radius+nudge)
	atom_charges.append(1)
	orbital_funcs.append([orbital_1s])

	print("Setup atom positions")

	generate_orbitals(atom_positions, atom_charges, orbital_funcs, atomic_orbitals, total_potential)
	print("Generated orbitals")

	eigs = calculateMOcoeffs(atomic_orbitals, total_potential)
	print("Calculated MO coeffs")

	print("MO Coefficients:", eigs.eigenvectors.transpose()[np.argsort(eigs.eigenvalues)])
	print("Eigenvalues:", np.sort(np.real(eigs.eigenvalues)) / charge_e)

	energies = np.real(np.sort(eigs.eigenvalues))
	totalEnergy = groundStateEnergy(energies, atom_positions, atom_charges)
	print(f"Ground State Energy: {totalEnergy / charge_e} eV")
	output_data_table[iteration][0] = totalEnergy


	plotMOs(atomic_orbitals, eigs, quantile=0.5, num_cols=3)




#############
# Iteration
#############

iterations = 1

output_data_table = np.zeros((iterations, 5))
labels = ["Bonding MO Energy", "Antibonding MO Energy", "Nuclear Repulsion", "Electron Repulsion (n/a)", "Total Energy"]

spacing_array = np.linspace(1.43, 1.48, iterations)

for i in range(iterations):
	params = {
		"spacing": spacing_array[i]
	}
	run(i, params, output_data_table)

print(output_data_table)
print(f"Lowest Total Energy: {np.min(output_data_table[:, 4]) / charge_e} eV at index {np.argmin(output_data_table[:, 4])}, spacing {spacing_array[np.argmin(output_data_table[:, 4])]} bohr radii")

plt.plot(spacing_array, output_data_table / charge_e)
plt.legend(labels)
plt.show()

