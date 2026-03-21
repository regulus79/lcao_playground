from psi_tools import *

def generate_orbitals(atom_poses, atom_charges, orbital_funcs, atomic_orbitals_array, potential):
	for i, pos in enumerate(atom_poses):
		for func in orbital_funcs[i]:
			atomic_orbitals_array.append(psiFromFunc(specifc_func(func, *pos, atom_charges[i]), normalized = True))
		potential += psiFromFunc(specifc_func(potentialWell, *pos, atom_charges[i]))
		#potential += psiFromFunc(specifc_func(smoothPotentialWell, *pos, atom_charges[i]))

def calculateMOcoeffs(atomic_orbitals, potential):
	H = np.zeros((len(atomic_orbitals), len(atomic_orbitals)), dtype = complex)
	S = np.zeros((len(atomic_orbitals), len(atomic_orbitals)), dtype = complex)
	print("Calculating inner products")
	for i in range(len(atomic_orbitals)):
		for j in range(len(atomic_orbitals)):
			H[i][j] = np.sum(np.conjugate(atomic_orbitals[i]) * apply_hamiltonian(potential, atomic_orbitals[j]))
			S[i][j] = np.sum(np.conjugate(atomic_orbitals[i]) * atomic_orbitals[j])
	A = np.linalg.inv(S)@H
	print("Setup matrix")
	return np.linalg.eig(A)

def getMOs(atomic_orbitals, eigs):
	MOs = []
	for i in range(eigs.eigenvectors.shape[1]):
		MO = np.sum((np.transpose(atomic_orbitals) * eigs.eigenvectors[:, i]).transpose(), axis = 0)
		MO /= np.sum(MO.conjugate() * MO)**0.5
		MOs.append(MO)
	return MOs

# ignores electron-electron repulsion
# assumes neutral molecules, #nuc = #e-
def groundStateEnergy(orbital_energies, atom_positions, atom_charges):
	orbital_energies = np.sort(orbital_energies)
	num_electrons = sum(atom_charges)
	total_orbital_energy = 0
	for i in range(num_electrons):
		total_orbital_energy += orbital_energies[i // 2]
	nuclei_replusion_energy = 0
	for i in range(len(atom_charges)):
		for j in range(len(atom_charges)):
			if i < j:
				dist = np.sum((np.array(atom_positions[i]) - np.array(atom_positions[j]))**2)**0.5
				nuclei_replusion_energy += atom_charges[i]*atom_charges[j]*charge_e**2 / (4 * math.pi * epsilon0 * dist)
	return total_orbital_energy + nuclei_replusion_energy

def plotMOs(atomic_orbitals, eigs, quantile = 0.5, num_cols = 3):
	fig = plt.figure()
	num_plots = eigs.eigenvalues.shape[0]
	num_rows = math.ceil(num_plots / num_cols)
	plot_index = 1
	for i in np.argsort(eigs.eigenvalues):
		ax = setupAxes(f"E = {np.real(eigs.eigenvalues[i]) / charge_e} eV", num_rows, num_cols, plot_index)
		plotPsi(np.sum((np.transpose(atomic_orbitals) * eigs.eigenvectors[:, i]).transpose(), axis = 0), ax, quantile)
		plot_index += 1
	plt.show()
