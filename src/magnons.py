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
from copy import deepcopy
import numpy as np
from pymatgen.core.structure import Structure
import matplotlib.pyplot as plt

import src.constants as const


def prepare_YIG_structure(filename):
    struct = Structure.from_file(filename)
    struct.remove_species(["Y", "O"])
    for i in range(12):
        struct.sites[i].species = "Fe1"
    for j in range(12, 20):
        struct.sites[j].species = "Fe2"
    return struct


def get_T(D):
    N = len(D) // 2
    g = np.diag([1]*N + [-1]*N)

    # We want D = K^dagger K whereas numpy uses K K^dagger
    K = np.conjugate(np.linalg.cholesky(D)).T
    L, U = np.linalg.eig(K @ g @ np.conjugate(K).T)

    # Arrange so that the first N are positive
    # And last N are negatives of the first N 
    # (see Eq. I4 in Tanner's Dissertation)
    sort_order = np.argsort(L)[::-1]
    sort_order = np.concatenate((sort_order[:N], sort_order[2*N:N-1:-1]))
    U = U[:, sort_order]
    L = np.diag(L[sort_order])

    # Now E is 1/2*diag([omega_1, omega_2, ..., omega_N,
    #                    omega_1, omega_2, ..., omega_N])
    # See Eq. (I4) in Tanner's Dissertation
    E = g @ L
    omega = 2*np.diag(np.real(E[:N])) 
    T = np.linalg.inv(K) @ U @ np.sqrt(E)

    return omega, T

def get_magnon_eig(hamiltonian, k, G):
    """
    hamiltonian: spin hamiltonian from rad tools
    k: k point in cartesian coordinates, units of eV
    G: reciprocal lattice vector in cartesian coordinates, units of eV
    """
    dispersion = MagnonDispersion(hamiltonian, phase_convention='tanner')

    n_atoms = len(hamiltonian.magnetic_atoms)  # Number of magnetic atoms
    # Atom positions in cartesian coordinates (units of 1/eV)
    xj = np.array(
        [
            hamiltonian.get_atom_coordinates(atom, relative=False)
            for atom in hamiltonian.magnetic_atoms
        ]
    )
    # Spins
    spins = np.array([atom.spin for atom in hamiltonian.magnetic_atoms])

    # The rj vectors in cartesian coordinates (dimensionless?)
    rj = dispersion.u

    #omega_nu_k = dispersion.omega(k)
    # omega_nu_k, Tk = solve_via_colpa(dispersion.h(k))
    # _, inv_Tk = solve_via_colpa(dispersion.h(-k)/2)
    # See Colpa Eq. 3.7 for this inversion trick
    # inv_Tk[:n, n:] *= -1
    # inv_Tk[n:, :n] *= -1
    # Tk = np.conjugate(inv_Tk).T
    # Tk = np.linalg.inv(Tk) --> Slower way

    # See Tanner's Disseration and RadTools doc for explanation of the 1/2
    omega_nu_k, Tk = get_T(dispersion.h(k)/2) 
    Uk_conj = np.conjugate(Tk[:n_atoms, :n_atoms]) # This is U_{j,nu,k})*
    V_minusk = np.conjugate(Tk[n_atoms:, :n_atoms]) # This is ((V_{j,nu,-k})*)*

    # Getting it with the other matrix since U and V are weird
    # Tk = get_T(dispersion.h(-k)/2)
    # Uk_conj = Tk[n:, n:] # This is (U_{j,nu,k})*
    # V_minusk = Tk[:n, n:] # This is (V_{j,nu,-k}

    # epsilon_nu_k_G = ( # This is wrong I think
    #    np.sqrt(spins / 2)[:, np.newaxis]
    #    * (
    #        np.tensordot(V_minusk, np.conjugate(rj), axes=1)
    #        + np.tensordot(Uk_conj, rj, axes=1)
    #    )
    #    * np.exp(1j * np.dot(G, xj.T)[:, np.newaxis])
    # )
    epsilon_nu_k_G = np.zeros((n_atoms, 3), dtype=complex)

    # If you use G and x in fractional coords, you need a factor of 2*pi
    # in the phase. Proof: the factor of 2*pi in the definition of b_{1,2,3}
    # This should be rewritten with numpy broadcasting
    #for nu in range(n_atoms):
    #    # Not sure if V and U are j,nu or nu,j...
    #    for j in range(n_atoms):
    #        epsilon_nu_k_G[nu, :] += (
    #            np.sqrt(spins[j] / 2)
    #            * (V_minusk[j, nu] * np.conjugate(rj[j]) + Uk_conj[j, nu] * rj[j])
    #            * np.exp(1j * np.dot(G, xj[j]))
    #        )
    prefactor = np.sqrt(spins / 2) * np.exp(1j * np.dot(xj, G))
    epsilon_nu_k_G = (prefactor[:, None] * V_minusk).T @ np.conjugate(rj) + (prefactor[:, None] * Uk_conj).T @ rj

    return omega_nu_k, epsilon_nu_k_G  # n, and nx3 array of complex numbers


def get_YIG_hamiltonian(filename):
    struct = prepare_YIG_structure(filename)
    site_type = lambda i: "d" if i < 12 else "a"
    # See blue notebook notes page 31 regarding why this is frac_coords
    # Even though documentation says it should be cartesian ("absolute")
    atoms = [
        Atom(f"Fe_{site_type(i)}", s.frac_coords, index=i + 1, spin=5 / 2)
        for i, s in enumerate(struct.sites)
    ]

    for atom in atoms:
        if atom.index < 13:
            atom.spin_direction = [0, 0, 1]
        else:
            atom.spin_direction = [0, 0, -1]

    # Convert everything to eV and inverse eV
    scaled_struct = deepcopy(struct)
    scaled_struct.scale_lattice(struct.volume * (const.Ang_To_inveV)**3)

    # Having to pass standardize = False every round because of inheritance is a bit absurd
    lattice = Lattice(np.round(scaled_struct.lattice.matrix, decimals=7), standardize=False)
    crystal = Crystal(lattice, atoms, standardize=False)
    hamiltonian = SpinHamiltonian(crystal, standardize=False)

    kelvin_to_eV = 8.617333262145 * 1e-5

    # From that paper cited by Tanner
    Jad = ExchangeParameter(iso=-40 * kelvin_to_eV)
    Jdd = ExchangeParameter(iso=-13.4 * kelvin_to_eV)
    Jaa = ExchangeParameter(iso=-3.8 * kelvin_to_eV)
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

    return hamiltonian


def get_YIG_neighbors(struct, neighbor_cutoff=5.5):
    """
    Returns the indices, distances, and R vectors of the nearest neighbors
    neighbor_cutoff: distance in Angstroms
    """
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


def print_YIG_neighbor_report(
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
                report_string += f"{i+1:2d} ({site_type(i)}),  {d:.3f} 1/eV, R = {R}\n"
                prev_d = d
    print(report_string)


def prepare_MVBT_structure(filename):
    struct = Structure.from_file(filename)
    struct.remove_species(["Bi", "Te"])
    # struct.sites[0].species = struct.sites[0].species + "1"
    # struct.sites[1].species = struct.sites[0].species + "2"
    return struct


def get_MVBT_neighbors(struct, neighbor_cutoff=25):
    indices = [{"inter": [], "intra": []} for i in range(struct.num_sites)]
    distances = [{"inter": [], "intra": []} for i in range(struct.num_sites)]
    R = [{"inter": [], "intra": []} for i in range(struct.num_sites)]
    z = [{"inter": [], "intra": []} for i in range(struct.num_sites)]
    for i, current_site in enumerate(struct.sites):
        neighbors = struct.get_neighbors(current_site, neighbor_cutoff)
        for n in neighbors:
            if n.index == i and n.coords[2] == current_site.coords[2]:
                # Neighbor is same kind of site and at same z value
                key = "intra"
            elif n.index != i and n.coords[2] != current_site.coords[2]:
                # Neighbor is different kind of site and different z value
                key = "inter"
            else:
                # This is a neighbor of the same kind of site but from a different layer
                # We neglect these
                continue
            indices[i][key].append(n.index)
            R[i][key].append(n.image)
            distances[i][key].append(n.distance(current_site, [0, 0, 0]))
            z[i][key].append(n.coords[2])
        distance_sort = {k: np.argsort(distances[i][k]) for k in ["inter", "intra"]}
        for k in ["inter", "intra"]:
            indices[i][k] = np.array(indices[i][k])[distance_sort[k]]
            distances[i][k] = np.array(distances[i][k])[distance_sort[k]]
            R[i][k] = np.array(R[i][k])[distance_sort[k]]
            z[i][k] = np.array(z[i][k])[distance_sort[k]]
            neighbor_class = -1
            prev_d = 0
            new_indices, new_distances, new_R, new_z = [[]], [], [[]], [[]]
            for j, d, RR, zz in zip(indices[i][k], distances[i][k], R[i][k], z[i][k]):
                if ~np.isclose(d, prev_d):
                    neighbor_class += 1
                    new_indices.append([])
                    new_distances.append(d)
                    new_R.append([])
                    new_z.append([])
                new_indices[neighbor_class].append(j)
                new_R[neighbor_class].append(RR)
                new_z[neighbor_class].append(zz)
                prev_d = d
            # Get rid of that empty list because this is an atrocious implementation
            indices[i][k] = new_indices[:-1]
            distances[i][k] = new_distances[:-1]
            R[i][k] = new_R[:-1]
            z[i][k] = new_z[:-1]

    return indices, distances, R, z


def print_MVBT_neighbor_report(
    struct,
):
    neighbor_indices, neighbor_distances, neighbor_R, neighbor_z = get_MVBT_neighbors(struct)
    report_string = ""

    for current_index, (current_site, indices, distances, Rs, zs) in enumerate(
        zip(struct.sites, neighbor_indices, neighbor_distances, neighbor_R, neighbor_z)
    ):
        print(f"Site {current_index+1} (z = {current_site.coords[2]})")
        for k in ["inter", "intra"]:
            print(f"  {k}layer neighbors")
            print(f"  {'*' * len(f'{k}layer neighbors')}")
            for neighbor_class, (i_class, d, R_class, z_class) in enumerate(
                zip(indices[k], distances[k], Rs[k], zs[k])
            ):
                print(f"    {neighbor_class+1}-Nearest Neighbor")
                for i, R, z in zip(i_class, R_class, z_class):
                    print(f"      Site {i} at z = {z:+.3f},  d = {d:.3f} A, R = {R}")
    print(report_string)

def get_MVBT_hamiltonian(filename, J, spin_direction,spin):
    struct = prepare_MVBT_structure(filename)
    atoms = [
        Atom("V", s.frac_coords, index=i, spin=spin)
        for i, s in enumerate(struct.sites)
    ]
    for a in atoms:
        a.spin_direction = ((-1) ** a.index) * np.array(spin_direction)

    # Convert everything to eV and inverse eV
    scaled_struct = deepcopy(struct)
    scaled_struct.scale_lattice(struct.volume * (const.Ang_To_inveV)**3)
    lattice = Lattice(np.round(scaled_struct.lattice.matrix, decimals=7), standardize=False)
    crystal = Crystal(lattice, atoms, standardize=False)
    hamiltonian = SpinHamiltonian(crystal, standardize=False)

    J = {
        "inter": [ExchangeParameter(iso=JJ) for JJ in J["inter"]],
        "intra": [ExchangeParameter(iso=JJ) for JJ in J["intra"]],
    }

    hamiltonian.double_counting = True # Really not sure about this...
    hamiltonian.spin_normalized = False  # I don't think it's normalized??
    hamiltonian.factor = -1/2 # Really not sure about this eitehr...

    # Add all pairs
    all_indices, _, all_R, _ = get_MVBT_neighbors(struct)
    for current_index, (neighbor_indices, neighbor_R) in enumerate(zip(all_indices, all_R)):
        # Now this gives the index of the current site, 
        # and dictionaries with the inter/intra layer neighbors
        for k in ["inter", "intra"]:
            for neighbor_class, (class_indices, class_R) in enumerate(
                zip(neighbor_indices[k], neighbor_R[k])
            ):
                # now each of these have neighbors in the same class (i.e., same distance away)
                for neighbor_index, R in zip(class_indices, class_R):
                    hamiltonian.add_bond(
                        atoms[current_index], atoms[neighbor_index], tuple(R), J[k][neighbor_class]
                    )
    return hamiltonian