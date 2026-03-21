import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib import colors
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


bohr_radius = 5.291772e-11
h_bar = 6.626070e-34 / (2*math.pi)
mass_e = 9.109384e-31
charge_e = 1.602176e-19
epsilon0 = 8.854188e-12

lattice_shape = (32, 32, 32)
lattice_data_size = lattice_shape[0]*lattice_shape[1]*lattice_shape[2]
lattice_radius = 10 * bohr_radius
length_per_step = 2*lattice_radius / lattice_shape[0]


def latticeConfig(shape, radius):
	global lattice_shape, lattice_data_size, lattice_radius, length_per_step
	lattice_shape = shape
	lattice_data_size = lattice_shape[0]*lattice_shape[1]*lattice_shape[2]
	lattice_radius = radius
	length_per_step = 2*lattice_radius / lattice_shape[0]
def latticeShape():
	return lattice_shape
def latticeRadius():
	return lattice_radius
def latticeLengthPerStep():
	return length_per_step


def psiFromFunc(func, normalized = False):
	newPsi = np.zeros(lattice_shape, dtype = complex)
	for x in range(lattice_shape[0]):
		for y in range(lattice_shape[1]):
			for z in range(lattice_shape[2]):
				newPsi[x,y,z] = func((x - lattice_shape[0]/2) * length_per_step, (y - lattice_shape[1]/2) * length_per_step, (z - lattice_shape[2]/2) * length_per_step)
	if normalized:
		newPsi /= np.sum(newPsi.conjugate() * newPsi)**0.5
	return newPsi


def plotPsi(psi, ax, quantile = 0.5):
	prob_density = np.absolute(psi)**2
	prob_density /= np.sum(prob_density)
	level = np.quantile(prob_density, 1 - quantile, weights = prob_density, method = "inverted_cdf")
	verts, faces, normals, values = measure.marching_cubes(prob_density, level)
	facecolors = [colors.hsv_to_rgb((np.angle(psi[tuple(np.round(np.mean(verts[face], axis = 0)).astype(int))]) / (2*math.pi) + 0.5, 1, 1)) for face in faces]
	verts *= length_per_step / bohr_radius # dividing by bohr radius to keep things not super small
	verts -= 0.5 * np.array([length_per_step * lattice_shape[0], length_per_step * lattice_shape[1], length_per_step * lattice_shape[2]]) / bohr_radius
	mesh = Poly3DCollection(verts[faces], shade=True, facecolors = facecolors)
	ax.add_collection3d(mesh)

def setupAxes(title, rows = 1, cols = 1, index = 1):
	ax = plt.subplot(rows, cols, index, projection = "3d")
	ax.title.set_text(title)
	ax.set_xlim3d(-lattice_radius / bohr_radius, lattice_radius / bohr_radius)
	ax.set_ylim3d(-lattice_radius / bohr_radius, lattice_radius / bohr_radius)
	ax.set_zlim3d(-lattice_radius / bohr_radius, lattice_radius / bohr_radius)
	ax.set_aspect("equal")
	return ax



def apply_potential(V, psi):
	return psi * V

def apply_laplacian(psi):
	newPsi = np.zeros_like(psi, dtype = complex)
	for x in range(lattice_shape[0]):
		for y in range(lattice_shape[1]):
			for z in range(lattice_shape[2]):
				if x <= 0 or x >= lattice_shape[0] - 1 or y <= 0 or y >= lattice_shape[1] - 1 or z <= 0 or z >= lattice_shape[2] - 1:
					continue
				delta = length_per_step
				xCurvature = ((psi[x + 1,y,z] - psi[x,y,z])/delta - (psi[x,y,z] - psi[x - 1,y,z])/delta) / (delta)
				yCurvature = ((psi[x,y + 1,z] - psi[x,y,z])/delta - (psi[x,y,z] - psi[x,y - 1,z])/delta) / (delta)
				zCurvature = ((psi[x,y,z + 1] - psi[x,y,z])/delta - (psi[x,y,z] - psi[x,y,z - 1])/delta) / (delta)
				newPsi[x,y,z] = (xCurvature + yCurvature + zCurvature)
	return newPsi

def apply_hamiltonian(V, psi):
	return -apply_laplacian(psi) * h_bar**2 / (2 * mass_e) + apply_potential(V, psi) * charge_e


def orbital_1s(x,y,z, charge):
	r = (x*x+y*y+z*z)**0.5 / bohr_radius * charge
	return math.exp(-r)

def orbital_2s(x,y,z, charge):
	r = (x*x+y*y+z*z)**0.5 / bohr_radius * charge
	return math.exp(-r / 2) * (2 - r)

def orbital_2p_z(x,y,z, charge):
	r = (x*x+y*y+z*z)**0.5 / bohr_radius * charge
	z_scaled = z / bohr_radius
	return math.exp(-r/2) * z_scaled
def orbital_2p_right(x,y,z, charge):
	r = (x*x+y*y+z*z)**0.5 / bohr_radius * charge
	phi = math.atan2(x,y)
	xy_scaled = (x*x+y*y)**0.5 / bohr_radius * charge
	return math.exp(-r/2) * xy_scaled/2 * np.exp(1j * phi)
def orbital_2p_left(x,y,z, charge):
	r = (x*x+y*y+z*z)**0.5 / bohr_radius * charge
	phi = math.atan2(x,y)
	xy_scaled = (x*x+y*y)**0.5 / bohr_radius * charge
	return math.exp(-r/2) * xy_scaled/2 * np.exp(-1j * phi)

def potentialWell(x,y,z, charge):
	scale = charge_e / (4*math.pi*epsilon0) * charge
	return -scale * (x*x + y*y + z*z)**-0.5 if x*x + y*y + z*z > 0 else 0

def smoothPotentialWell(x,y,z, charge):
	scale = charge_e / (4*math.pi*epsilon0) * charge
	smooth_factor = 0.1 * bohr_radius
	return -scale * ((x*x + y*y + z*z)**-0.5 + smooth_factor) if x*x + y*y + z*z > 0 else 0

def specifc_func(func, x0,y0,z0, charge):
	return lambda x,y,z: func(x - x0, y - y0, z - z0, charge)
