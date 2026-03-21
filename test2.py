import numpy as np

def M_linear(size,diag,offdiag):
	return diag * np.diag(np.ones((size,))) + offdiag * (np.diag(np.ones((size-1,)), 1) + np.diag(np.ones((size-1,)), -1))

def M_ring(size, diag, offdiag):
	return M_ring(size, diag, offdiag) + offdiag*(np.diag(np.ones((1,)), size-1) + np.diag(np.ones((1,)), -size+1))


# x^TV^THVx + L(1 - x^TV^TVx)
# V^THVx - LV^TVx = 0
# V^THVx = LV^TVx
# V^THVx = LV^TVx
# (V^TV)^-1V^THVx = Lx
#
# (H - ES)x = 0

size = 2

H = M_linear(size, -23, 0)
S = M_linear(size, 1, 0)

print(f"H: {H}")
print(f"S: {S}")

A = np.linalg.inv(S)@H

print(f"A: {A}")

eigs = np.linalg.eig(A)

print(eigs)

for i in range(eigs.eigenvalues.shape[0]):
	print(f"--- Val: {eigs.eigenvalues[i]}, Vec: {eigs.eigenvectors[:, i]}")
	#print(H - eigs.eigenvalues[i]*S)
	#print((H - eigs.eigenvalues[i]*S) @ eigs.eigenvectors[:, i])