# Calculate ground truth values (Young's modulus and yield stress) from stress-
# strain curves. Stress-strain curves should be loaded from Zenodo to run this.

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt


#SET = 'test'
SET = 'tr_val'

LOAD_PATH = '../../molecular_dynamics/stress_strain_curves/'  # make sure this folder is loaded from Zenodo
SAVE_PATH = f'{SET}_set/'
EL_RANGE = 0.0015
OFFSET = 0.005


def calc_young_modulus():

    strain = np.load(f'{LOAD_PATH}strain.npy')

    step = strain[1000]/1000
    end_idx = int(EL_RANGE/step)

    stresses = np.load(f'{LOAD_PATH}stress_shifted_{SET}_set.npy')
    stresses = ma.masked_invalid(stresses)

    k, b= ma.polyfit(strain[:end_idx], stresses[:, :end_idx].T, deg=1)

    np.save(f'{SAVE_PATH}moduluses_{EL_RANGE}.npy', k)
    

def calc_off_set_yield():

    strain = np.load(f'{LOAD_PATH}strain.npy')

    step = strain[1000]/1000
    end_idx = int(EL_RANGE/step)


    stresses = np.load(f'{LOAD_PATH}stress_shifted_{SET}_set.npy')
    stresses = ma.masked_invalid(stresses)

    yield_strains = []
    yield_stresses = []
    for i, stress in enumerate(stresses):
        print(i)
        k, b= ma.polyfit(strain[:end_idx], stress[:end_idx], deg=1)
        yield_strain, yield_stress = offset_yield(strain, stress, k, b)
        yield_strains.append(yield_strain)
        yield_stresses.append(yield_stress)

    yield_strains = np.array(yield_strains)
    yield_stresses = np.array(yield_stresses)

    np.save(f'{SAVE_PATH}yield_strains_{OFFSET}.npy', yield_strains)
    np.save(f'{SAVE_PATH}yield_stress_{OFFSET}.npy', yield_stresses)


def offset_yield(strain, stress, k, b):
    

    line_y = k*(strain-OFFSET)

    yield_strain = 0
    yield_stress = 0

    for x, y_stress, y_line in zip(strain, moving_average(stress), line_y):
        
        if y_line >= y_stress:
            yield_strain = x
            yield_stress = y_stress
            break

        
    """plt.plot(strain[:150000], stress[:150000])
    plt.plot(strain[:150000], moving_average(stress)[:150000], 'k')
    step = strain[1]
    end_idx = int(EL_RANGES[0]/step)
    plt.plot([0, strain[end_idx]], [b, k*strain[end_idx]+b], 'b')
    line_y = k*(strain[:130000]-OFFSET)
    plt.plot(strain[:130000], line_y, 'b')
    plt.plot([yield_strain, yield_strain], [0, 8], 'k')
    plt.plot([0, 0.1], [yield_stress, yield_stress], 'k')
    plt.ylim([0,5])
    plt.xlim([0,0.045])
    plt.grid()
    plt.show()"""
    
    return yield_strain, yield_stress


def moving_average(a, n=301, remove_masked=False):

    if remove_masked:
        start=0
        while ma.is_masked(a[start]):
            start+=1
        a[:start] = 0

        end=160000-1
        while ma.is_masked(a[end]):
            end-=1
        a[end+1:] = a[end]

    if n%2!=1:
        raise ValueError

    start = a[0]*np.ones(int((n-1)/2))
    end = a[-1]*np.ones(int((n-1)/2))

    x = ma.hstack((start, a, end))

    ret = ma.cumsum(x, dtype=float)
    ret[n:] -= ret[:-n]
    ret = ret[n - 1:] / n

    #ret[nan_idx] = np.nan
    return ret


calc_young_modulus()
calc_off_set_yield()