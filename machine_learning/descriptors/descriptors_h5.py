# Generates the engineered descriptor fields (grain boundary atom
# density and lattice orientation) used as inputs to the CNN. Doesn't include
# very surfaces of the pillars, since they are considered as grain boundaries
# by the polyhedral template matching algortithm.

# Prior to running this, local lattice orientations must be calculated and saved
# with Ovito's polyhedral template matching algorithm. See readme.md for
# instructions.

import numpy as np
from scipy.spatial.transform import Rotation
import h5py

res = 16
#res = 32

coord_min = 4.8 # edges are considered grain boundaries by the polyhedral template matching algortithm so they are not included

limits='constant'

rotation='quaternion'
#rotation='euler_angle'

#euler_type='intrinsic'
euler_type='extrinsic'

rot_n_channels = {'quaternion':4, 'euler_angle': 3}

tr_N = 4000
val_N = 500
test_N = 500
N_tot = tr_N+val_N+test_N

if rotation == 'quaternion':
    rotation_fname = 'quaternion'
elif rotation == 'euler_angle':
    rotation_fname = f'euler_angle_{euler_type}'

rot_file = h5py.File(f'{rotation_fname}_{res}.h5', 'w')
rot_file.create_dataset('training_set', (tr_N,res,res,2*res,rot_n_channels[rotation]), dtype=np.float32)
rot_file.create_dataset('validation_set', (val_N,res,res,2*res,rot_n_channels[rotation]), dtype=np.float32)
rot_file.create_dataset('test_set', (test_N,res,res,2*res,rot_n_channels[rotation]), dtype=np.float32)

grain_file = h5py.File(f'grain_boundary_not_normalized_{res}.h5', 'w')
grain_file.create_dataset('training_set', (tr_N,res,res,2*res,1), dtype=np.uint8)
grain_file.create_dataset('validation_set', (val_N,res,res,2*res,1), dtype=np.uint8)
grain_file.create_dataset('test_set', (test_N,res,res,2*res,1), dtype=np.uint8)


grain_boundary_sum = np.zeros((res, res, 2*res, 1), dtype=np.uint32)

xmin=coord_min
ymin=coord_min
zmin=coord_min

xmax=100-coord_min
ymax=100-coord_min
zmax=200-coord_min

xlen=100-2*coord_min
ylen=100-2*coord_min
zlen=200-2*coord_min

for conf in range(N_tot):

    print(conf)

    dist = np.full((res,res,2*res),-1.0)
    pillar_grain = np.zeros((res, res, 2*res, 1), dtype=np.uint8)
    pillar_rot = np.zeros((res, res, 2*res, rot_n_channels[rotation]), dtype=np.float32)

    if conf < val_N:
        save_set = 'validation_set'
        c = conf
    elif conf < tr_N+val_N:
        save_set = 'training_set'
        c = conf-val_N
    else:
        save_set = 'test_set'
        c = conf-val_N-tr_N

    rot_set = rot_file[save_set]
    grain_set = grain_file[save_set]

    # file generated with Ovito's polyhedral template matching
    if save_set =='test_set':
        fname = f'raw/descriptors_test_set.{c}'
    else:
        fname = f'raw/descriptors_tr_val_set.{conf}'

    with open(fname) as f:
        lines = f.readlines()
    
    for line in lines[2:]:
        
        word = line.split()

        lattice_type = word[7]

        x = float(word[0])
        y = float(word[1])
        z = float(word[2])

        if x<xmin or y<ymin or z<zmin or x>xmax or y>ymax or z>zmax:
            continue

        nx = int((x-xmin)/(xlen/res))
        ny = int((y-ymin)/(ylen/res))
        nz = int((z-zmin)/(zlen/(2*res)))

        # sidestepping error when x==xmax
        if nx == res:
            nx = res-1
        if ny == res:
            ny = res-1
        if nz == 2*res:
            nz = 2*res - 1

        x0 = xmin + (nx + 0.5) * (xlen/res)
        y0 = ymin + (ny + 0.5) * (ylen/res)
        z0 = zmin + (nz + 0.5) * (zlen/(2*res))

        dist_sq = (x - x0)**2 + (y - y0)**2 + (z - z0)**2
        if (dist[nx][ny][nz] == -1 or dist_sq < dist[nx][ny][nz]):

            dist[nx][ny][nz] = dist_sq

            if lattice_type=='BCC':
                rot_x = float(word[3])
                rot_y = float(word[4])
                rot_z = float(word[5])
                rot_w = float(word[6])

                if rotation == 'quaternion':
                    pillar_rot[nx,ny,nz, :] = np.array((rot_x, rot_y, rot_z, rot_w), dtype=np.float32)
                
                elif rotation == 'euler_angle':
                    rot = Rotation.from_quat((rot_x, rot_y, rot_z, rot_w))

                    euler_sequence_string={'intrinsic': 'XYZ', 'extrinsic': 'xyz'}
                    rot_euler = rot.as_euler(seq=euler_sequence_string[euler_type])
                    rot_euler = np.asarray(rot_euler)

                    pillar_rot[nx][ny][nz] = rot_euler


            else:
                pillar_rot[nx][ny][nz] = np.zeros(rot_n_channels[rotation])

            
        if lattice_type != 'BCC':
            pillar_grain[nx][ny][nz][0] = pillar_grain[nx][ny][nz][0] + 1

    rot_set[c] = pillar_rot
    grain_set[c] = pillar_grain

    if save_set != 'test_set':
        grain_boundary_sum += pillar_grain

grain_boundary_mean = grain_boundary_sum/(tr_N+val_N)

grain_file_normalized = h5py.File(f'grain_boundary_{res}.h5', 'w')
grain_file_normalized.create_dataset('training_set', (tr_N,res,res,2*res,1), dtype=np.float32)
grain_file_normalized.create_dataset('validation_set', (val_N,res,res,2*res,1), dtype=np.float32)
grain_file_normalized.create_dataset('test_set', (test_N,res,res,2*res,1), dtype=np.float32)

sum_squared_diff = np.zeros((res, res, 2*res, 1))
for pillar in grain_file['validation_set']:
    pillar = pillar.astype(np.float64)
    sum_squared_diff += (pillar-grain_boundary_mean)**2
for pillar in grain_file['training_set']:
    pillar = pillar.astype(np.float64)
    sum_squared_diff += (pillar-grain_boundary_mean)**2

grain_boundary_std = np.sqrt(sum_squared_diff / (tr_N+val_N))

for set_name in ('validation_set', 'training_set', 'test_set'):
    grain_set = grain_file[set_name]
    grain_set_normalized = grain_file_normalized[set_name]

    for c, pillar in enumerate(grain_set):
        pillar = pillar.astype(np.float64)
        pillar_normalized = (pillar-grain_boundary_mean)/grain_boundary_std
        grain_set_normalized[c] = pillar_normalized


rot_file.close()
grain_file.close()
grain_file_normalized.close()
