# Finds the optimal c11, c12, and c44 values for the undefined voxels using a
# grid search

import numpy as np


def invert_arrays(arr):
    
    original_shape = arr.shape
    reshaped = arr.reshape(-1, 6, 6)
    
    inverted = np.linalg.inv(reshaped)

    return inverted.reshape(original_shape)


def trim_C_field(C_field, peelz, peelxy):

    n = C_field.shape[0]
    mask_inner = np.zeros_like(C_field, dtype=bool)
    mask_inner[:, peelxy:res-peelxy, peelxy:res-peelxy, peelz:2*res-peelz, :, :] = True
    C_field = C_field[mask_inner].reshape(n, res-2*peelxy, res-2*peelxy, 2*res-2*peelz, 6, 6)
    return C_field

def trim_local_modulus_field(field, peelz, peelxy):

    n = field.shape[0]
    mask_inner = np.zeros_like(field, dtype=bool)
    mask_inner[:, peelxy:res-peelxy, peelxy:res-peelxy, peelz:2*res-peelz] = True
    field = field[mask_inner].reshape(n, res-2*peelxy, res-2*peelxy, 2*res-2*peelz)
    return field


def insert_to_zeros(C_field, zero_mask, c11, c12, c44):
    C_field[zero_mask, 0, 0] = c11
    C_field[zero_mask, 1, 1] = c11
    C_field[zero_mask, 2, 2] = c11

    C_field[zero_mask, 0, 1] = c12
    C_field[zero_mask, 0, 2] = c12
    C_field[zero_mask, 1, 2] = c12
    C_field[zero_mask, 1, 0] = c12
    C_field[zero_mask, 2, 0] = c12
    C_field[zero_mask, 2, 1] = c12

    C_field[zero_mask, 3, 3] = c44
    C_field[zero_mask, 4, 4] = c44
    C_field[zero_mask, 5, 5] = c44

    return C_field

def C_from_cii(c11, c12, c44):

    C = np.zeros((6,6))

    C[0, 0] = c11
    C[1, 1] = c11
    C[2, 2] = c11

    C[0, 1] = c12
    C[0, 2] = c12
    C[1, 2] = c12
    C[1, 0] = c12
    C[2, 0] = c12
    C[2, 1] = c12

    C[3, 3] = c44
    C[4, 4] = c44
    C[5, 5] = c44

    return C

def C_field_nonzero_mean(C_field):
    
    mask = np.any(C_field != 0, axis=(4, 5))
    
    nonzero_counts = np.sum(mask, axis=(1, 2, 3))
    
    arr_masked = np.where(mask[:, :, :, :, np.newaxis, np.newaxis], C_field, np.nan)
    
    mean_nonzero = np.nanmean(arr_masked, axis=(1, 2, 3))
    
    return mean_nonzero, nonzero_counts

def E_field_nonzero_harmonic_mean(E_field):
    
    mask = E_field != 0
    
    nonzero_counts = np.sum(mask, axis=(1, 2, 3))
    
    arr_masked = np.where(mask, E_field, np.nan)
    
    harmonic_mean_nonzero = nonzero_counts/np.nansum(1/arr_masked, axis=(1, 2, 3))
    
    return harmonic_mean_nonzero, nonzero_counts


def Ez_from_C(c11, c12, c44):
    s11 = (c11 + c12) / ((c11 - c12)*(c11 + 2*c12))

    return 1/s11

def E_reuss_func(E_local):
    return 1/np.sum(1/E_local, axis=(1,2,3))*np.prod(E_local.shape[1:])

def r_2(y_true, y_pred):
    RSS =  np.sum(np.square(y_true - y_pred))
    TSS = np.sum(np.square(y_true - np.mean(y_true)))
    return ( 1 - (RSS/TSS) )


#res = 16
res = 32
#avg_type = 'voigt'
avg_type = 'hill'

global_modulus_tr_val_set = np.load('../../machine_learning/ground_truth/tr_val_set/moduluses_0.0015.npy')
global_modulus_test_set = np.load('../../machine_learning/ground_truth/test_set/moduluses_0.0015.npy')
E_gt_tr_val_set_normalized = (global_modulus_tr_val_set-np.mean(global_modulus_tr_val_set))/np.std(global_modulus_tr_val_set)

peelz = {16:3, 32:6}
peelxy = 0

if res==16:
    C_test_set = np.load(f'C_field_test_set_{res}.npy')
    C_tr_val_set = np.load(f'C_field_tr_val_set_{res}.npy')
    C_test_set = trim_C_field(C_test_set, peelz[res], peelxy)
    C_tr_val_set = trim_C_field(C_tr_val_set, peelz[res], peelxy)
    # We precompute the mean of defined voxels and number undefined voxels to speed up computations
    C_mean_nonzero_tr_val_set, nonzero_counts_tr_val_set = C_field_nonzero_mean(C_tr_val_set)
    C_mean_nonzero_test_set, nonzero_counts_test_set = C_field_nonzero_mean(C_test_set)
else:
    C_mean_nonzero_tr_val_set = np.load(f'C_mean_nonzero_tr_val_set_{res}.npy')
    C_mean_nonzero_test_set = np.load(f'C_mean_nonzero_test_set_{res}.npy')
    nonzero_counts_tr_val_set = np.load(f'nonzero_counts_tr_val_set_{res}.npy')
    nonzero_counts_test_set = np.load(f'nonzero_counts_test_set_{res}.npy')

# local modulus field E(r)=1/s_33(r), which is used in the hill average
local_modulus_field_tr_val_set = np.load(f'../../machine_learning/Grad-CAM/local_modulus_field_tr_val_set_{res}.npy')
local_modulus_field_test_set = np.load(f'../../machine_learning/Grad-CAM/local_modulus_field_test_set_{res}.npy')
local_modulus_field_test_set = trim_local_modulus_field(local_modulus_field_test_set, peelz[res], peelxy)
local_modulus_field_tr_val_set = trim_local_modulus_field(local_modulus_field_tr_val_set, peelz[res], peelxy)
E_harm_mean_nonzero_tr_val_set, local_modulus_nonzero_counts_tr_val_set = E_field_nonzero_harmonic_mean(local_modulus_field_tr_val_set)
E_harm_mean_nonzero_test_set, local_modulus_nonzero_counts_test_set = E_field_nonzero_harmonic_mean(local_modulus_field_test_set)

n_voxels = np.prod(local_modulus_field_tr_val_set.shape[1:4])

zero_mask_local_modulus_test_set = local_modulus_field_test_set==0

# voigt_wide_search_16 and 32
search_ress = [64, 16, 4, 1]
c11_seach_bounds = [0, 2000]
c12_seach_bounds = [0, 2000]
c44_seach_bounds = [0, 2000]

# hill_wide_search_16 and 32
search_ress = [64, 16, 4, 1]
c11_seach_bounds = [1, 2000]
c12_seach_bounds = [0, 2000]
c44_seach_bounds = [0, 2000]

# We start with a sparse grid search and increase the resolution iteratively
for search_res in search_ress:
    c11s = np.arange(c11_seach_bounds[0], c11_seach_bounds[1], search_res)
    c12s = np.arange(c12_seach_bounds[0], c12_seach_bounds[1], search_res)
    c44s = np.arange(c44_seach_bounds[0], c44_seach_bounds[1], search_res)

    r2_grid = np.zeros((len(c11s), len(c12s), len(c44s)))
    E_pred_grid = np.zeros((len(c11s), len(c12s), len(c44s), global_modulus_tr_val_set.size))

    for i_c11, c11 in enumerate(c11s):
        print()
        for i_c12, c12 in enumerate(c12s):
            print()
            for i_c44, c44 in enumerate(c44s):
                
                C_at_undefined = C_from_cii(c11, c12, c44)
                C_mean = nonzero_counts_tr_val_set[:, np.newaxis, np.newaxis]/n_voxels * C_mean_nonzero_tr_val_set + (n_voxels-nonzero_counts_tr_val_set[:, np.newaxis, np.newaxis])/n_voxels * C_at_undefined

                S_mean = invert_arrays(C_mean)

                if avg_type == 'voigt':
                    E_estimate = 1/S_mean[:,2,2]
                
                if avg_type == 'hill':
                    E_replace = Ez_from_C(c11, c12, c44)

                    E_reuss = n_voxels/(local_modulus_nonzero_counts_tr_val_set/E_harm_mean_nonzero_tr_val_set + (n_voxels-local_modulus_nonzero_counts_tr_val_set)/E_replace)

                    s33_reuss = 1/E_reuss
                    s33_voigt = S_mean[:,2,2]
                    s33_hill = 0.5*(s33_voigt+s33_reuss)

                    E_estimate = 1/s33_hill

                r2 = r_2(E_gt_tr_val_set_normalized, (E_estimate-np.mean(E_estimate))/np.std(E_estimate))
                print(r2)

                r2_grid[i_c11, i_c12, i_c44] = r2
                E_pred_grid[i_c11, i_c12, i_c44] = E_estimate


    max_idx_flat = np.argmax(r2_grid)
    max_idx = np.unravel_index(max_idx_flat, r2_grid.shape)
    c11_best_idx, c12_best_idx, c44_best_idx = max_idx
    c11_best = c11s[c11_best_idx]
    c12_best = c12s[c12_best_idx]
    c44_best = c44s[c44_best_idx]
    r2_best = np.max(r2_grid)
    E_pred_tr_val_set = E_pred_grid[c11_best_idx, c12_best_idx, c44_best_idx]

    print('r2:',r2_best)
    print('c values:', c11_best, c12_best, c44_best)

    next_search_res = int(search_res/4)
    scan = 2
    c11_seach_bounds = [c11_best-scan*next_search_res, c11_best+scan*next_search_res+1]
    c12_seach_bounds = [c12_best-scan*next_search_res, c12_best+scan*next_search_res+1]
    c44_seach_bounds = [c44_best-scan*next_search_res, c44_best+scan*next_search_res+1]


C_at_undefined = C_from_cii(c11_best, c12_best, c44_best)
C_mean_test_set = nonzero_counts_test_set[:, np.newaxis, np.newaxis]/n_voxels * C_mean_nonzero_test_set + (n_voxels-nonzero_counts_test_set[:, np.newaxis, np.newaxis])/n_voxels * C_at_undefined

S_mean_test_set = invert_arrays(C_mean_test_set)
if avg_type == 'voigt':
    E_pred_test_set = 1/S_mean_test_set[:,2,2]

if avg_type == 'hill':
    E_replace = Ez_from_C(c11_best, c12_best, c44_best)
    local_modulus_field_test_set[zero_mask_local_modulus_test_set] = E_replace

    E_reuss = E_reuss_func(local_modulus_field_test_set)
    s33_reuss = 1/E_reuss
    s33_voigt = S_mean_test_set[:,2,2]
    s33_hill = 0.5*(s33_voigt+s33_reuss)

    E_pred_test_set = 1/s33_hill

r2_test_set = r_2((global_modulus_test_set-np.mean(global_modulus_tr_val_set))/np.std(global_modulus_tr_val_set),
            (E_pred_test_set-np.mean(E_pred_tr_val_set))/np.std(E_pred_tr_val_set))

print('r2 test set:', r2_test_set)
E_replace = Ez_from_C(c11_best, c12_best, c44_best)
print('E replace', E_replace)

np.savez(f'{avg_type}_wide_search_{res}.npz', r2_test_set=r2_test_set, r2_tr_val_set=r2_best, r2_grid=r2_grid, E_pred_test_set=E_pred_test_set, E_pred_tr_val_set=E_pred_tr_val_set, c11=c11_best, c12=c12_best, c44=c44_best, E_replace=E_replace)
