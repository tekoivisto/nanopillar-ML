# Generates the atom positions descriptor field used as an input to the CNN. 

import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import multiprocessing as mp
import h5py

res = 128

lattice_constant = 3.300101

coord_min = lattice_constant
coord_min = 0
coord_min_array = np.array([coord_min, coord_min, 2*coord_min])
side_length = 100-2*coord_min
voxel_len = side_length/res

val_n = 500

cut_off_radius = lattice_constant
cut_off_radius_voxel = ceil(cut_off_radius/voxel_len)

load_path = '../../molecular_dynamics/nanopillars/'  # make sure this folder is loaded from Zenodo
save_path = './'

sigma_gaussian = 1.0375332832336426  # value from sigma_search.py
exp_multiplier = -0.5/sigma_gaussian**2
def brigthness_function_gaussian(dist_sq):
    return np.exp(dist_sq * exp_multiplier)


def process_single_pillar(pillar_atom_coords, i):
    print(i)

    pillar_result = np.zeros((res, res, 2*res), dtype=np.float32)

    for atom_coord in pillar_atom_coords:
        if np.isnan(atom_coord[0]):
            break
        x, y, z = atom_coord
        atom_coord_int = np.floor((atom_coord-coord_min_array)/voxel_len).astype(int)
        nx, ny, nz = atom_coord_int

        # Create a 3D grid for the voxel neighborhood
        nx_range = np.arange(max(0, nx - cut_off_radius_voxel), min(res, nx + cut_off_radius_voxel + 1))
        ny_range = np.arange(max(0, ny - cut_off_radius_voxel), min(res, ny + cut_off_radius_voxel + 1))
        nz_range = np.arange(max(0, nz - cut_off_radius_voxel), min(2*res, nz + cut_off_radius_voxel + 1))
        nx0, ny0, nz0 = np.meshgrid(nx_range, ny_range, nz_range, indexing='ij')

        # Calculate distances and apply brightness function

        x0 = coord_min + (nx0 + 0.5) * voxel_len
        y0 = coord_min + (ny0 + 0.5) * voxel_len
        z0 = 2*coord_min + (nz0 + 0.5) * voxel_len
        dst_sq = (x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2
        
        brightness = brigthness_function_gaussian(dst_sq)

        # Update pillars with brightness values where they are greater than current values
        pillar_result[nx_range[:, None, None], ny_range[None, :, None], nz_range[None, None, :]] = np.maximum(
            pillar_result[nx_range[:, None, None], ny_range[None, :, None], nz_range[None, None, :]],
            brightness
        )
    
    return pillar_result

def main():


    h5file = h5py.File(f'{save_path}pillars_brigthness_function_{sigma_gaussian:.2f}_{res}.h5', 'w')
    
    mean = 0
    std = 0
    for set_, fname in zip(('tr_val', 'test'), ('atom_coords_tr_val_set.npy', 'atom_coords_test_set.npy')):
        print(set_)
        atom_positions = np.load(load_path + fname)
        N_pillars = atom_positions.shape[0]
        pillars = np.zeros((N_pillars, res, res, 2*res),dtype=np.float32)

        # Use multiprocessing to process each pillar
        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.starmap(process_single_pillar, [(pillar, i) for i, pillar in enumerate(atom_positions[:N_pillars])])

        # Combine results
        for i, result in enumerate(results):
            pillars[i] = result

        if set_ == 'tr_val':
            mean = np.mean(pillars)
            std = np.std(pillars)
            print('mean', mean)
            print('std', std)
            print('shape', pillars.shape)
        
        pillars = (pillars-mean)/std
        pillars = pillars.astype(np.float32)

        if set_ == 'tr_val':
            h5file.create_dataset('validation_set', data=pillars[:val_n])
            h5file.create_dataset('training_set', data=pillars[val_n:])
        elif set_ == 'test':
            h5file.create_dataset('test_set', data=pillars)

    h5file.close()


if __name__ == "__main__":
    main()
