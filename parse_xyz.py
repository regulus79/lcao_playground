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


def table_effective_charges(atom_charges): # this is so bad.
	effective_charge_table = [# adapted from wikipedia http://en.wikipedia.org/wiki/Effective_nuclear_charge
		0, 
		1, # H
		1.688, # He
		1.279, # Li
		1.912, # Be
		2.576, # B
		3.217, # C
		3.847, # N
		4.492 # O
	]
	tmp = []
	for charge in atom_charges:
		tmp.append(effective_charge_table[charge])
	return tmp