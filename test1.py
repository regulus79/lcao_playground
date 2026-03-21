import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib import colors
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

orbital_shape = (32, 32, 32)
orbital_data_size = orbital_shape[0]*orbital_shape[1]*orbital_shape[2]
orbital_radius = 13


def psiFromFunc(func):
	newPsi = np.zeros(orbital_shape, dtype = complex)
	for x in range(orbital_shape[0]):
		for y in range(orbital_shape[1]):
			for z in range(orbital_shape[2]):
				newPsi[x,y,z] = func(x * 2*orbital_radius / orbital_shape[0] - orbital_radius, y * 2*orbital_radius / orbital_shape[1] - orbital_radius, z * 2*orbital_radius / orbital_shape[2] - orbital_radius)
	return newPsi


def plotPsi(psi, quantile = 0.5):
	prob_density = np.absolute(psi)**2
	prob_density /= np.sum(prob_density)
	level = np.quantile(prob_density, 1 - quantile, weights = prob_density, method = "inverted_cdf")
	verts, faces, normals, values = measure.marching_cubes(prob_density, level)
	facecolors = [colors.hsv_to_rgb((np.angle(psi[tuple(np.round(np.mean(verts[face], axis = 0)).astype(int))]) / (2*math.pi) + 0.5, 1, 1)) for face in faces]
	verts *= 2*orbital_radius / orbital_shape[0]
	verts -= np.array([orbital_radius, orbital_radius, orbital_radius])
	mesh = Poly3DCollection(verts[faces], shade=True, facecolors = facecolors)
	plt.gca().add_collection3d(mesh)

def setupFigure(title):
	fig = plt.figure()
	fig.suptitle(title)
	ax = fig.add_subplot(projection = "3d")
	ax.set_xlim3d(-orbital_radius, orbital_radius)
	ax.set_ylim3d(-orbital_radius, orbital_radius)
	ax.set_zlim3d(-orbital_radius, orbital_radius)
	ax.set_aspect("equal")



def potential(V, psi):
	return psi * V

def laplacian(psi):
	newPsi = np.zeros_like(psi)
	for x in range(orbital_shape[0]):
		for y in range(orbital_shape[1]):
			for z in range(orbital_shape[2]):
				if x <= 0 or x >= orbital_shape[0] - 1 or y <= 0 or y >= orbital_shape[1] - 1 or z <= 0 or z >= orbital_shape[2] - 1:
					continue
				delta = orbital_radius / orbital_shape[0]
				xCurvature = ((psi[x + 1,y,z] - psi[x,y,z])/delta - (psi[x,y,z] - psi[x - 1,y,z])/delta) / (2*delta)
				yCurvature = ((psi[x,y + 1,z] - psi[x,y,z])/delta - (psi[x,y,z] - psi[x,y - 1,z])/delta) / (2*delta)
				zCurvature = ((psi[x,y,z + 1] - psi[x,y,z])/delta - (psi[x,y,z] - psi[x,y,z - 1])/delta) / (2*delta)
				newPsi[x,y,z] = xCurvature + yCurvature + zCurvature
	return newPsi




def orbital_1s(x,y,z):
	r = (x*x+y*y+z*z)**0.5
	return math.exp(-r)

def orbital_2s(x,y,z):
	r = (x*x+y*y+z*z)**0.5
	return math.exp(-r / 2) * (2 - r)

def orbital_2p_z(x,y,z):
	r = (x*x+y*y+z*z)**0.5
	costheta = z / r if r > 0 else 0
	return math.exp(-r/2) * costheta * r
def orbital_2p_right(x,y,z):
	r = (x*x+y*y+z*z)**0.5
	phi = math.atan2(x,y)
	sintheta = math.sqrt(1 - (z / r)**2) if r > 0 else 0
	return math.exp(-r/2) * sintheta * -r/2 * np.exp(1j * phi)
def orbital_2p_left(x,y,z):
	r = (x*x+y*y+z*z)**0.5
	phi = math.atan2(x,y)
	sintheta = math.sqrt(1 - (z / r)**2) if r > 0 else 0
	return math.exp(-r/2) * sintheta * r/2 * np.exp(-1j * phi)

def potentialWell(x,y,z):
	return -(x*x + y*y + z*z)**-0.5 if x*x + y*y + z*z > 0 else 0

def offset_func(func, x0,y0,z0):
	return lambda x,y,z: func(x - x0, y - y0, z - z0)


atom_positions = []
atomic_orbitals = []
total_potential = np.zeros(orbital_shape, dtype = complex)
spacing = 3.5
#for i in range(6):
#	atom_positions.append(((i - 3) * spacing, 0 * 0.3 * math.cos(i * math.pi) * spacing, 0))
for i in range(6):
	theta = i/6 * math.pi*2
	atom_positions.append((math.cos(theta) * spacing * 6 / math.pi, math.sin(theta) * spacing * 6 / math.pi, 0))

for pos in atom_positions:
	total_potential += psiFromFunc(offset_func(potentialWell, *pos))
	atomic_orbitals.append(psiFromFunc(offset_func(orbital_2p_z, *pos)))

H = np.zeros((len(atomic_orbitals), len(atomic_orbitals)), dtype = complex)
S = np.zeros((len(atomic_orbitals), len(atomic_orbitals)), dtype = complex)

for i in range(len(atomic_orbitals)):
	for j in range(len(atomic_orbitals)):
		H[i][j] = np.sum(np.conjugate(atomic_orbitals[i]) * (-laplacian(atomic_orbitals[j]) + potential(total_potential, atomic_orbitals[j])))
		S[i][j] = np.sum(np.conjugate(atomic_orbitals[i]) * atomic_orbitals[j])

print(H)
print(S)

A = np.linalg.inv(S)@H

eigs = np.linalg.eig(A)

print(eigs)

quantile = 0.5

for i in range(eigs.eigenvalues.shape[0]):
	setupFigure(f"E = {eigs.eigenvalues[i]}")
	plotPsi(np.sum((np.transpose(atomic_orbitals) * eigs.eigenvectors[:, i]).transpose(), axis = 0), quantile)

plt.show()

exit()

atom1pos = (1,0,0)
atom2pos = (-1,0,0)


mypotential = psiFromFunc(offset_func(potentialWell, *atom1pos)) * 6
mypotential2 = psiFromFunc(offset_func(potentialWell, *atom2pos)) * 6

mypsi = psiFromFunc(offset_func(orbital_2p_z, *atom1pos))
mypsi /= np.sum(np.absolute(mypsi)**2)**0.5

mypsi2 = psiFromFunc(offset_func(orbital_2p_z, *atom2pos))
mypsi2 /= np.sum(np.absolute(mypsi2)**2)**0.5

totalpotential = mypotential + mypotential2

hamiltonain1 = -laplacian(mypsi) + potential(totalpotential, mypsi)
hamiltonain2 = -laplacian(mypsi2) + potential(totalpotential, mypsi2)


print(np.sum(mypsi * mypsi))
print(np.sum(mypsi2 * mypsi2))
print(np.sum(mypsi * mypsi2))
print(np.sum(mypsi * hamiltonain1))
print(np.sum(mypsi2 * hamiltonain2))
print(np.sum(mypsi2 * hamiltonain1))

setupFigure("psi1^2")
plotPsi(mypsi, quantile)
setupFigure("psi2^2")
plotPsi(mypsi2, quantile)

setupFigure("combined")

combined = mypsi + mypsi2
combined /= np.sum(np.absolute(combined)**2)**0.5

plotPsi(combined, quantile)

#setupFigure("potentials")
#plotPsi(mypotential)
#plotPsi(mypotential2)

plt.show()

