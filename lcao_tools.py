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
	num_inner_products = len(atomic_orbitals)**2
	num_calculated = 0
	for i in range(len(atomic_orbitals)):
		for j in range(len(atomic_orbitals)):
			H[i][j] = np.sum(np.conjugate(atomic_orbitals[i]) * apply_hamiltonian(potential, atomic_orbitals[j]))
			S[i][j] = np.sum(np.conjugate(atomic_orbitals[i]) * atomic_orbitals[j])
			num_calculated += 1
			print(f"Calculating inner products... {num_calculated}/{num_inner_products}", end = "\r")
	print("")
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

def homo_lumo_gap(orbital_energies, num_electrons):
	orbital_energies = np.sort(orbital_energies)
	homo_index = (num_electrons - 1) // 2
	homo = orbital_energies[homo_index]
	lumo = orbital_energies[homo_index + 1]
	return lumo - homo

def plotAtomPositions(ax, atom_positions):
	colors = ["black", "pink", "yellow", "grey", "blue", "red", "purple", "green", "brown", "orange"] # aaaaa this is bad
	for i in range(len(atom_positions)):
		ax.scatter(atom_positions[i][0] / bohr_radius, atom_positions[i][1] / bohr_radius, atom_positions[i][2] / bohr_radius, color = colors[0]) # TODO color

def plotMOs(atomic_orbitals, eigs, atom_positions, num_electrons, quantile = 0.5, num_cols = 3):
	fig = plt.figure()
	num_plots = eigs.eigenvalues.shape[0]
	num_rows = math.ceil(num_plots / num_cols)
	plot_index = 1
	for i in np.argsort(eigs.eigenvalues):
		ax = setupAxes(f"E = {np.real(eigs.eigenvalues[i]) / charge_e} eV", num_rows, num_cols, plot_index)
		plotPsi(np.sum((np.transpose(atomic_orbitals) * eigs.eigenvectors[:, i]).transpose(), axis = 0), ax, quantile)
		plotAtomPositions(ax, atom_positions)
		plot_index += 1
	plt.show()

def plotHOMOLUMO(atomic_orbitals, eigs, atom_positions, num_electrons, quantile = 0.5):
	orbital_energies = np.sort(np.real(eigs.eigenvalues))
	homo_index = (num_electrons - 1) // 2
	lumo_index = homo_index + 1
	fig = plt.figure(figsize = (11, 6))
	ax1 = setupAxes(f"HOMO {orbital_energies[homo_index] / charge_e} eV", 1, 2, 1)
	plotPsi(np.sum((np.transpose(atomic_orbitals) * eigs.eigenvectors[:, homo_index]).transpose(), axis = 0), ax1, quantile)
	plotAtomPositions(ax1, atom_positions)
	ax2 = setupAxes(f"LUMO {orbital_energies[lumo_index] / charge_e} eV", 1, 2, 2)
	plotPsi(np.sum((np.transpose(atomic_orbitals) * eigs.eigenvectors[:, lumo_index]).transpose(), axis = 0), ax2, quantile)
	plotAtomPositions(ax2, atom_positions)
	plt.show()

def plotHOMOLUMO_more(atomic_orbitals, eigs, atom_positions, num_electrons, quantile = 0.5):
	orbital_energies = np.sort(np.real(eigs.eigenvalues))
	homo_index = (num_electrons - 1) // 2
	lumo_index = homo_index + 1
	fig = plt.figure(figsize = (18, 6))
	if homo_index-1 >= 0 and homo_index-1 < len(eigs.eigenvalues):
		ax1 = setupAxes(f"HOMO-1 {orbital_energies[homo_index-1] / charge_e} eV", 1, 4, 1)
		plotPsi(np.sum((np.transpose(atomic_orbitals) * eigs.eigenvectors[:, homo_index - 1]).transpose(), axis = 0), ax1, quantile)
		plotAtomPositions(ax1, atom_positions)
	if homo_index >= 0 and homo_index < len(eigs.eigenvalues):
		ax2 = setupAxes(f"HOMO {orbital_energies[homo_index] / charge_e} eV", 1, 4, 2)
		plotPsi(np.sum((np.transpose(atomic_orbitals) * eigs.eigenvectors[:, homo_index]).transpose(), axis = 0), ax2, quantile)
		plotAtomPositions(ax2, atom_positions)
	if lumo_index >= 0 and lumo_index < len(eigs.eigenvalues):
		ax3 = setupAxes(f"LUMO {orbital_energies[lumo_index] / charge_e} eV", 1, 4, 3)
		plotPsi(np.sum((np.transpose(atomic_orbitals) * eigs.eigenvectors[:, lumo_index]).transpose(), axis = 0), ax3, quantile)
		plotAtomPositions(ax3, atom_positions)
	if lumo_index+1 >= 0 and lumo_index+1 < len(eigs.eigenvalues):
		ax4 = setupAxes(f"LUMO+1 {orbital_energies[lumo_index+1] / charge_e} eV", 1, 4, 4)
		plotPsi(np.sum((np.transpose(atomic_orbitals) * eigs.eigenvectors[:, lumo_index + 1]).transpose(), axis = 0), ax4, quantile)
		plotAtomPositions(ax4, atom_positions)
	plt.show()