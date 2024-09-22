# Finds a value for sigma which is used in atom_positions.py
# brigthness_function_gaussian. The value is chosen such that the mean of the
# descriptor field (prior to normalizing) is 0.5. Bisection method is used to
# find the sigma value.

import numpy as np
import matplotlib.pyplot as plt
from math import ceil

#res = 64
res = 128

lattice_constant = 3.300101

coord_min = 0
coord_min_array = np.array([coord_min, coord_min, 2*coord_min])
side_length = 100-2*coord_min
voxel_len = side_length/res

val_n = 500

cut_off_radius = lattice_constant
cut_off_radius_voxel = ceil(cut_off_radius/voxel_len)
print(cut_off_radius_voxel)

load_path = '../../molecular_dynamics/nanopillars/'  # make sure this folder is loaded from Zenodo

portion = 0.1
sigma_gaussian = lattice_constant/(2*np.sqrt(2)*np.sqrt(np.log(1/portion)))
exp_multiplier = -0.5/sigma_gaussian**2
def brigthness_function_gaussian(dist, sigma):
    return np.exp(-0.5 * dist**2 / sigma**2)


def process_single_pillar(pillar_atom_coords, i):
    print(i)

    pillar_result = np.full((res, res, 2*res),np.inf, dtype=float)

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
        
        #brightness = brigthness_function_gaussian(dst_sq)
        brightness = np.sqrt(dst_sq)

        # Update pillars with brightness values where they are greater than current values
        pillar_result[nx_range[:, None, None], ny_range[None, :, None], nz_range[None, None, :]] = np.minimum(
            pillar_result[nx_range[:, None, None], ny_range[None, :, None], nz_range[None, None, :]],
            brightness
        )
    
    return pillar_result


def f(dists, sigma):

    brightnesses = brigthness_function_gaussian(dists, sigma)
    return np.mean(brightnesses)-0.5

def bisection(a, b, tolerance, max_iterations, dists):
    if f(dists, a) * f(dists, b) > 0:
        print("Bisection method fails.")
        print(f(dists, a), a)
        print(f(dists, b), b)
        return None

    iteration = 1
    while (b - a) / 2 > tolerance and iteration <= max_iterations:
        midpoint = (a + b) / 2
        if f(dists, midpoint) == 0:
            return midpoint
        elif f(dists, a) * f(dists, midpoint) < 0:
            b = midpoint
        else:
            a = midpoint
        iteration += 1

        print('iteretion', iteration, 'complete. Midpoint:', midpoint)

    return (a + b) / 2

if __name__ == "__main__":

    atom_positions = np.load(load_path + 'atom_coords_test_set.npy')

    pillars_all = np.array([])
    for i in range(20):
        pillar = process_single_pillar(atom_positions[i], i)
        leave_out = int(2*lattice_constant/voxel_len)
        pillar = pillar[leave_out:res-leave_out, leave_out:res-leave_out, leave_out:2*res-leave_out]

        pillar = pillar.flatten()

        pillars_all = np.concatenate((pillars_all, pillar))
    

    print(pillars_all.shape)
    print(pillars_all.mean())

    print('infs:', np.count_nonzero(np.isinf(pillars_all)))
    pillars_all[np.isinf(pillars_all)] = 3
    plt.hist(pillars_all, 5000)
    l=lattice_constant/2
    plt.xlabel('distance to closest atom from voxel center')
    plt.ylabel('number of voxels')

    sigma = bisection(0.5, 3, 0.00001, 1000, pillars_all)
    print('final sigma:', sigma) # res 128 1.0375332832336426, res 64 1.0375385284423828

    plt.figure()
    pillars_all = np.sort(pillars_all)
    plt.hist(pillars_all, 1000, density=True)
    plt.plot(pillars_all, 1-np.arange(0, pillars_all.size)/pillars_all.size)
    #l = np.sqrt(3)*lattice_constant/4
    plt.plot([l, l], [0,1])

    x = np.linspace(0, 3, 1000)
    plt.plot(x, brigthness_function_gaussian(x, sigma=sigma))


    plt.xlabel('distance to closest atom from voxel center')
    plt.ylabel('1 - cumulative portion of voxels')

    plt.figure()
    plt.hist(brigthness_function_gaussian(pillars_all, sigma), 1000)
    plt.xlabel('brightness')
    plt.ylabel('num voxels')

    plt.show()