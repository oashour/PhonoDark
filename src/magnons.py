def prepare_YIG_structure(filename):
    struct = Structure.from_file(filename)
    struct.remove_species(["Y", "O"])
    for i in range(12):
        struct.sites[i].species = "Fe1"
    for j in range(12, 20):
        struct.sites[j].species = "Fe2"
    return struct


from radtools import (
    SpinHamiltonian,
    ExchangeParameter,
    Atom,
    Crystal,
    Lattice,
    MagnonDispersion,
    Kpoints,
    solve_via_colpa,
)
import numpy as np
from pymatgen.core.structure import Structure
import matplotlib.pyplot as plt


def get_rj(hamiltonian):
    n = len(hamiltonian.magnetic_atoms)
    spin_directions = [atom.spin_direction for atom in hamiltonian.magnetic_atoms]
    cross = [np.cross(s, [0, 0, 1]) for s in spin_directions]
    dot = [np.dot(s, [0, 0, 1]) for s in spin_directions]
    CPM = lambda x: np.array([-np.cross(x, np.eye(3)[i, :]) for i in range(3)])
    R = [
        -1 * np.eye(3)
        if dot[i] == -1
        else np.eye(3)
        + CPM(cross[i])
        + (CPM(cross[i]) @ CPM(cross[i])) / (1 + np.dot(spin_directions[i], [0, 0, 1]))
        for i in range(n)
    ]
    return [[R[j][alpha, 0] + 1j * R[j][alpha, 1] for alpha in range(3)] for j in range(n)]


def get_magnon_eigenvectors(hamiltonian, k, G):
    dispersion = MagnonDispersion(hamiltonian)

    n = len(hamiltonian.magnetic_atoms)  # Number of magnetic atoms
    k = np.array([0, 0, 0])  # Test vector for now
    xj = np.array([atom.position for atom in hamiltonian.magnetic_atoms])
    spins = np.array([atom.spin for atom in hamiltonian.magnetic_atoms])

    rj = get_rj(hamiltonian)

    omegak, Tk = solve_via_colpa(dispersion.h(k))
    Uk = Tk[:n, :n]
    V_minusk = np.conjugate(Tk[n:, :n])
    epsilon_nu_k_G = np.sqrt(spins / 2)[:, np.newaxis] * np.tensordot(
        V_minusk, np.conjugate(rj), axes=1
    ) + np.tensordot(np.conjugate(Uk), rj, axes=1) * np.exp(1j * np.dot(G, xj.T)[:, np.newaxis])

    return epsilon_nu_k_G


def get_YIG_hamiltonian(filename):
    struct = prepare_YIG_structure(filename)
    site_type = lambda i: "d" if i < 12 else "a"
    atoms = [
        Atom(f"Fe_{site_type(i)}", s.frac_coords, index=i + 1, spin=5 / 2)
        for i, s in enumerate(struct.sites)
    ]

    for atom in atoms:
        if atom.index < 13:
            atom.spin_direction = [0, 0, +1]
        else:
            atom.spin_direction = [0, 0, -1]

    crystal = Crystal(Lattice(np.round(struct.lattice.matrix, decimals=7)), atoms)
    hamiltonian = SpinHamiltonian(crystal)

    kelvin_to_meV = 8.617333262145 * 1e-2

    Jad = ExchangeParameter(iso=-40 * kelvin_to_meV)
    Jdd = ExchangeParameter(iso=-13.4 * kelvin_to_meV)
    Jaa = ExchangeParameter(iso=-3.8 * kelvin_to_meV)
    J = {"aa": Jaa, "dd": Jdd, "ad": Jad, "da": Jad}

    hamiltonian.double_counting = False
    hamiltonian.spin_normalized = False  # I don't think it's normalized??
    hamiltonian.factor = -2

    # Add all pairs
    all_indices, all_distances, all_R = get_YIG_neighbors(struct)
    for current_index, (neighbor_indices, neighbor_R) in enumerate(zip(all_indices, all_R)):
        for neighbor_index, R in zip(neighbor_indices, neighbor_R):
            parameter_type = site_type(current_index) + site_type(neighbor_index)
            hamiltonian.add_bond(
                atoms[current_index], atoms[neighbor_index], tuple(R), J[parameter_type]
            )


def get_YIG_neighbors(struct, neighbor_cutoff=5.5):
    indices = []
    distances = []
    R = []
    for current_site in struct.sites:
        neighbors = struct.get_neighbors(current_site, neighbor_cutoff)
        neighbor_indices = [n.index for n in neighbors]
        neighbor_distance = [n.distance(current_site) for n in neighbors]
        neighbor_R = [n.frac_coords - n.to_unit_cell().frac_coords for n in neighbors]
        distance_sort = np.argsort(neighbor_distance)
        neighbor_indices = np.array(neighbor_indices)[distance_sort]
        neighbor_distance = np.array(neighbor_distance)[distance_sort]
        neighbor_R = np.array(neighbor_R)[distance_sort]
        indices.append(neighbor_indices)
        distances.append(neighbor_distance)
        R.append(neighbor_R)
        # Save results

    return indices, distances, R


def print_neighbor_report(
    struct,
):
    neighbor_indices, neighbor_distances, neighbor_R = get_YIG_neighbors(struct)
    report_string = ""
    site_type = lambda i: "d" if i < 12 else "a"
    prev_d = 0
    for current_index in range(struct.num_sites):
        current_neighbors = "Nearest Neighbors"
        report_string += "*********************************\n"
        report_string += f"*** Current Site: {current_index+1} ({site_type(current_index)})\n"
        for i, d, R in zip(
            neighbor_indices[current_index],
            neighbor_distances[current_index],
            neighbor_R[current_index],
        ):
            with np.printoptions(formatter={"float": "{:+0.0f}".format}):
                if ~np.isclose(d, prev_d):
                    report_string += current_neighbors + "\n"
                    report_string += "-" * len(current_neighbors) + "\n"
                    current_neighbors = "Next " + current_neighbors
                report_string += f"{i+1:2d} ({site_type(i)}),  {d:.3f} A, R = {R}\n"
                prev_d = d
    print(report_string)
