"""
    Compute the DM - phonon scattering rate for a general effective operator.

    Necessary input files:
        - DFT input files:
            - POSCAR - equilibrium positions of atoms
            - FORCE_SETS - second order force constants
            (optional) BORN - Born effective charges

        - inputs/
            - material/
                - (material name)/
                    (material_info).py - set the material parameters such as nucleon numbers and spins
            - numerics/
                (numerics_info).py - list of numerics parameters
            - physics_model/
                (physics_model_info).py - parameters defining the scattering potential


    Created by : Tanner Trickle, Zhengkang Zhang
"""

from mpi4py import MPI
import numpy as np
import phonopy
import os
import sys
import optparse

import src.constants as const
import src.parallel_util as put
import src.mesh as mesh
import src.phonopy_funcs as phonopy_funcs
import src.physics as physics
import src.hdf5_output as hdf5_output
from src.magnons import get_YIG_hamiltonian, get_MVBT_hamiltonian

#####

version = "1.1.0"

# initializing MPI
print('Getting comm')
comm = MPI.COMM_WORLD

# total number of processors
n_proc = comm.Get_size()

# processor if
proc_id = comm.Get_rank()

# ID of root process
root_process = 0

if proc_id == root_process:
    print('Got comm and size and everything.')
    print('------------')

if proc_id == root_process:

    print('\n--- Dark Matter - Phonon Scattering Rate Calculator ---\n')
    print('  version: '+version+'\n')
    print('  Running on '+str(n_proc)+' processors\n')
    print('------\n')

# Parse the input parameters ###
cwd = os.getcwd()

parser = optparse.OptionParser()
parser.add_option('-m', action="store", default="",
        help="Material info file. Contains the crystal lattice degrees of freedom for a given material.")
parser.add_option('-p', action="store", default="",
        help="Physics model input. Contains the coupling coefficients and defines which operators enter the scattering potential.")
parser.add_option('-n', action="store", default="",
        help="Numerics input. Sets the parameters used for the integration over momentum space and the input and output files.")

options_in, args = parser.parse_args()

options = vars(options_in)

def import_file(full_name, path):
    """
        Import a python module from a path. 3.4+ only.
        Does not call sys.modules[full_name] = path
    """
    from importlib import util

    spec = util.spec_from_file_location(full_name, path)
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

if options['m'] != '' and options['p'] != '' and options['n'] != '':

    # import modules
    material_input = options['m']
    physics_model_input = options['p']
    numerics_input = options['n']

    mat_input_mod_name = os.path.splitext(os.path.basename(material_input))[0]
    phys_input_mod_name = os.path.splitext(os.path.basename(physics_model_input))[0]
    num_input_mod_name = os.path.splitext(os.path.basename(numerics_input))[0]

    mat_mod = import_file(mat_input_mod_name, os.path.join(cwd, material_input))
    phys_mod = import_file(phys_input_mod_name, os.path.join(cwd, physics_model_input))
    num_mod = import_file(num_input_mod_name, os.path.join(cwd, numerics_input))

    if proc_id == root_process:

        print('Inputs :\n')
        print('  Material :\n')
        print('    Name : '+mat_mod.material+'\n')
        print('    DFT supercell grid size : '+str(mat_mod.mat_properties_dict['supercell_dim'])+'\n')
        print('    N : '+str(mat_mod.mat_properties_dict['N_list'])+'\n')
        print('    L.S : '+str(mat_mod.mat_properties_dict['L_S_list'])+'\n')
        print('    S : '+str(mat_mod.mat_properties_dict['S_list'])+'\n')
        print('    L : '+str(mat_mod.mat_properties_dict['L_list'])+'\n')
        print('    L.S_2 : '+str(mat_mod.mat_properties_dict['L_tens_S_list'])+'\n')
        print('  Physics Model :\n')
        print('    DM spin : '+str(phys_mod.dm_properties_dict['spin'])+'\n')
        print('    DM mass : '+str(phys_mod.dm_properties_dict['mass_list'])+'\n')
        print('    d log V / d log q : '+str(phys_mod.physics_parameters['power_V'])+'\n')
        print('    - d log F_med / d log q : '+str(phys_mod.physics_parameters['Fmed_power'])+'\n')
        print('    Time of day : '+str(phys_mod.physics_parameters['times'])+'\n')
        print('    Threshold : '+str(phys_mod.physics_parameters['threshold'])+' eV\n')
        print('    c coefficients : '+str(phys_mod.c_dict)+'\n')
        print('  Numerics :\n')
        print('    Energy bin width : '+str(num_mod.numerics_parameters['energy_bin_width'])+' eV\n')
        print('    N_a : '+str(num_mod.numerics_parameters['n_a'])+'\n')
        print('    N_b : '+str(num_mod.numerics_parameters['n_b'])+'\n')
        print('    N_c : '+str(num_mod.numerics_parameters['n_c'])+'\n')
        if num_mod.numerics_parameters['special_mesh']:
            print('    Special mesh : True\n')
        else:
            print('    power_a : '+str(num_mod.numerics_parameters['power_a'])+'\n')
            print('    power_b : '+str(num_mod.numerics_parameters['power_b'])+'\n')
            print('    power_c : '+str(num_mod.numerics_parameters['power_c'])+'\n')
        print('    q cut : '+str(num_mod.numerics_parameters['q_cut'])+'\n')
        print('    N_DW_x : '+str(num_mod.numerics_parameters['n_DW_x'])+'\n')
        print('    N_DW_y : '+str(num_mod.numerics_parameters['n_DW_y'])+'\n')
        print('    N_DW_z : '+str(num_mod.numerics_parameters['n_DW_z'])+'\n')
        print('------\n')

    job_list      = None
    job_list_recv = None

    if proc_id == root_process:

        print('Configuring calculation ...\n')

        # number of jobs to do
        num_masses    = len(phys_mod.dm_properties_dict['mass_list'])
        num_times     = len(phys_mod.physics_parameters['times'])

        num_jobs = num_masses*num_times
        print('  Total number of jobs : '+str(num_jobs))
        print()

        total_job_list = []

        for m in range(num_masses):
            for t in range(num_times):

                total_job_list.append([m, t])

        job_list = put.generate_job_list(n_proc, np.array(total_job_list))

    if proc_id == root_process:
        print('Going to scatter')
    # scatter the job list
    job_list_recv = comm.scatter(job_list, root=root_process)

    diff_rate_list   = []
    binned_rate_list = []
    total_rate_list  = []

    if proc_id == root_process:
        print('Done configuring calculation')

    material = mat_mod.material

    #hamiltonian = get_YIG_hamiltonian('symm_reduced_YIG.vasp')
    J = mat_mod.J_data
    spin_direction = [1, 0, 0]
    spin = 3/2
    hamiltonian = get_MVBT_hamiltonian('VBT_prim_mag_rot_z.vasp', J, spin_direction, spin)

    # run for the given jobs
    first_job = True
    for job in range(len(job_list_recv)):

        if (job_list_recv[job, 0] != -1 and job_list_recv[job, 1] != -1 ):

            job_id = job_list_recv[job]

            mass = phys_mod.dm_properties_dict['mass_list'][int(job_id[0])]
            time = phys_mod.physics_parameters['times'][int(job_id[1])]


            if first_job and proc_id == root_process:
                print('  Loading data from RAD-tools ...\n')

            if first_job and proc_id == root_process:

                print(f'    Number of atoms : {len(hamiltonian.magnetic_atoms)}\n')
                print(f'    Number of modes : {len(hamiltonian.magnetic_atoms)}\n')
                #print('    Atom masses : '+str(phonopy_params['atom_masses'])+'\n')

            #if not phys_mod.include_screen:
            #    if proc_id == root_process:
            #        print('  Include screen is FALSE. Setting the dielectric to the identity.\n')
            #        print()

            #    phonopy_params['dielectric'] = np.identity(3)

            if first_job and proc_id == root_process:
                print('  Done loading data from RAD-tools\n')

            # generate q mesh
            vE_vec = physics.create_vE_vec(time)

            delta = 2*phys_mod.physics_parameters['power_V'] - 2*phys_mod.physics_parameters['Fmed_power']

            [q_XYZ_list, jacob_list] = mesh.create_q_mesh(mass,
                                           phys_mod.physics_parameters['threshold'],
                                           vE_vec,
                                           num_mod.numerics_parameters,
                                           None,
                                           None,
                                           delta)

            # Beta testing a uniform q mesh for different calculations...

            # [q_XYZ_list, jacob_list] = mesh.create_q_mesh_uniform(mass,
            #                                phys_mod.physics_parameters['threshold'],
            #                                vE_vec,
            #                                num_mod.numerics_parameters,
            #                                phonon_file,
            #                                phonopy_params['atom_masses'],
            #                                delta,
            #                                q_red_to_XYZ = phonopy_params['recip_red_to_XYZ'],
            #                                mesh = [20, 20, 20]
            #                                )

            # Hamiltonian is already set up with lattice in inverse eV so this should be in eV
            recip_red_to_XYZ = np.transpose(hamiltonian.reciprocal_cell)
            #recip_XYZ_to_red = np.linalg.inv(recip_red_to_XYZ)
            k_XYZ_list = mesh.generate_k_XYZ_mesh_from_q_XYZ_mesh(q_XYZ_list, recip_red_to_XYZ)
            G_XYZ_list = mesh.get_G_XYZ_list_from_q_XYZ_list(q_XYZ_list, recip_red_to_XYZ)
            #k_red_list, G_red_list = mesh.get_kG_from_q_XYZ(q_XYZ_list, recip_red_to_XYZ)

            [diff_rate, binned_rate, total_rate] = physics.calc_diff_rates_magnon(mass,
                                                hamiltonian,
                                                q_XYZ_list,
                                                G_XYZ_list,
                                                k_XYZ_list,
                                                jacob_list,
                                                phys_mod.physics_parameters,
                                                vE_vec,
                                                num_mod.numerics_parameters,
                                                phys_mod.c_dict,
                                                mat_mod.mat_properties_dict,
                                                phys_mod.dm_properties_dict,
                                                phys_mod.c_dict_form)

            diff_rate_list.append([job_list_recv[job], np.real(diff_rate)])
            binned_rate_list.append([job_list_recv[job], np.real(binned_rate)])
            total_rate_list.append([job_list_recv[job], np.real(total_rate)])

            first_job = False

    print(f'********** Done computing rate on {proc_id}.')
    if proc_id == root_process:
        print('Done computing rate. Returning all data to root node to write.\n\n------\n')
    comm.Barrier()
    if proc_id == root_process:
        print('<------> Done on all processes, going to gather')

    # return data back to root
    all_diff_rate_list   = comm.gather(diff_rate_list, root=root_process)
    all_binned_rate_list = comm.gather(binned_rate_list, root=root_process)
    all_total_rate_list  = comm.gather(total_rate_list, root=root_process)

    # write to output file
    if proc_id == root_process:
        print('Done gathering!!!')
        print('----------')

        out_filename = os.path.join(
                num_mod.io_parameters['output_folder'],
                material+'_'+phys_input_mod_name+'_'+num_input_mod_name+num_mod.io_parameters['output_filename_extra']+'.hdf5')

        hdf5_output.hdf5_write_output(out_filename,
                                       num_mod.numerics_parameters,
                                       phys_mod.physics_parameters,
                                       phys_mod.dm_properties_dict,
                                       phys_mod.c_dict,
                                       all_total_rate_list,
                                       n_proc,
                                       material,
                                       all_diff_rate_list,
                                       all_binned_rate_list)

        print('Done writing rate.\n\n------\n')

else:

    if proc_id == root_process:
        print('ERROR:')
        print("\t- material info file\n"+
        "\t- physics model file\n"+
        "\t- numerics input file\n\n"+
        "must be present. These are added with the -m, -p, -n input flags. Add -h (--help) flags for more help.\n\n"+
        "See inputs/material/GaAs/GaAs_example.py, inputs/physics_model/dark_photon_example.py, inputs/numerics/standard.py for examples.")
