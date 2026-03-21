from psi_tools import *


latticeConfig((32,32,32), 8 * bohr_radius)

pos = (0.5 * bohr_radius*8/32, 0.5 * bohr_radius*8/32, 0.5 * bohr_radius*8/32)

potential = psiFromFunc(specifc_func(potentialWell, *pos, 1))

psi = psiFromFunc(specifc_func(orbital_2p_z, *pos, 1), normalized = True)

U_exp = np.sum(np.conjugate(psi) * apply_potential(potential, psi) * charge_e)
T_exp = np.sum(np.conjugate(psi) * apply_laplacian(psi) * h_bar**2 / (2 * mass_e))
H_exp = np.sum(np.conjugate(psi) * apply_hamiltonian(potential, psi))

print(U_exp / charge_e)
print(T_exp / charge_e)
print(H_exp / charge_e)

ax = setupAxes(f"")
plotPsi(psi + psiFromFunc(specifc_func(orbital_2s, *pos, 1), normalized = True), ax)
#ax = setupAxes(f"", 1, 4, 1)
#plotPsi(psiFromFunc(specifc_func(orbital_2s, *pos, 1), normalized = True), ax)
#ax = setupAxes(f"", 1, 4, 2)
#plotPsi(psiFromFunc(specifc_func(orbital_2p_z, *pos, 1), normalized = True), ax)
#ax = setupAxes(f"", 1, 4, 3)
#plotPsi(psiFromFunc(specifc_func(orbital_2p_left, *pos, 1), normalized = True), ax)
#ax = setupAxes(f"", 1, 4, 4)
#plotPsi(psiFromFunc(specifc_func(orbital_2p_right, *pos, 1), normalized = True), ax)

plt.show()