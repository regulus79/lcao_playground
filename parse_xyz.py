from psi_tools import *

from lcao_tools import *


atom_charges = {
	"H": 1,
	"C": 6,
	"N": 7,
	"O": 8
}

atom_orbitals = {
	""
}

def parse(file_text):
	positions = []
	charges = []
	for line in file_text.split("\n"):
		tokens = line.split()
		if len(tokens) < 4:
			continue
		atom_symbol = tokens[0]
		x = float(tokens[1]) * 1e-10
		y = float(tokens[2]) * 1e-10
		z = float(tokens[3]) * 1e-10
		charges.append(atom_charges[atom_symbol])
		positions.append(np.array([x, y, z]))
	return positions, charges


def default_orbital_functions(atom_charges):
	orbital_funcs = []
	for charge in atom_charges:
		if charge <= 2:
			orbital_funcs.append([orbital_1s])
		else:
			orbital_funcs.append([orbital_1s, orbital_2s, orbital_2p_z, orbital_2p_left, orbital_2p_right])
	return orbital_funcs

def valence_orbital_functions(atom_charges):
	orbital_funcs = []
	for charge in atom_charges:
		if charge <= 2:
			orbital_funcs.append([orbital_1s])
		else:
			orbital_funcs.append([orbital_2s, orbital_2p_z, orbital_2p_left, orbital_2p_right])
	return orbital_funcs

def z_orbital_functions(atom_charges):
	orbital_funcs = []
	for charge in atom_charges:
		if charge <= 2:
			orbital_funcs.append([])
		else:
			orbital_funcs.append([orbital_2p_z])
	return orbital_funcs


def valence_charges(atom_charges):
	tmp = []
	for charge in atom_charges:
		if charge <= 2:
			tmp.append(charge % 2)
		else:
			tmp.append((charge - 2) % 8)
	return tmp