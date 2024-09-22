# Generates the engineered descriptor fields (grain boundary atom
# density and lattice orientation). Similar to descriptors_h5.npy, but surfaces
# of the grid defining the field are aligned to surfaces of the pillar.

# Prior to running this, local lattice orientations must be calculated and saved
# with Ovito's polyhedral template matching algorithm. See readme.md for
# instructions.

import numpy as np
from scipy.spatial.transform import Rotation

res = 16
#res = 32

set_ = 'test'
N = 500
#set_ = 'tr_val'
#N = 4500



#limits = 'pillar'
limits='constant'

rotation='quaternion'
#rotation='euler_angle'

#euler_type='intrinsic'
euler_type='extrinsic'

rot_n_channels = {'quaternion':4, 'euler_angle': 3}



if rotation == 'quaternion':
    rotation_fname = 'quaternion'
elif rotation == 'euler_angle':
    rotation_fname = f'euler_angle_{euler_type}'

rot_all = np.zeros((N,res,res,2*res,rot_n_channels[rotation]), dtype=np.float32)
grain_all = np.zeros((N,res,res,2*res), dtype=np.uint32)


for conf in range(N):

    print(conf)

    dist = np.full((res,res,2*res),-1.0)

    # file generated with Ovito's polyhedral template matching
    fname = f'raw/descriptors_{set_}_set.{conf}'

    with open(fname) as f:
        lines = f.readlines()
    
    if limits == 'pillar':
        xmin = 50
        xmax = 50
        ymin = 50
        ymax = 50
        zmin = 50
        zmax = 50
        for line in lines[2:]:
            word = line.split()
            x = float(word[0])
            y = float(word[1])
            z = float(word[2])

            if x<xmin:
                xmin=x
            if x>xmax:
                xmax=x
            if y<ymin:
                ymin=y
            if y>ymax:
                ymax=y
            if z<zmin:
                zmin=z
            if z>zmax:
                zmax=z

        xlen=xmax-xmin
        ylen=ymax-ymin
        zlen=zmax-zmin

    elif limits == 'constant':

        xmin=0
        ymin=0
        zmin=0

        xlen=100
        ylen=100
        zlen=200

    for line in lines[2:]:
        
        word = line.split()

        lattice_type = word[7]

        x = float(word[0])
        y = float(word[1])
        z = float(word[2])

        if x<xmin or y<ymin or z<zmin or x>xlen or y>ylen or z>zlen:
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
                    rot_all[conf][nx][ny][nz][0] = rot_x
                    rot_all[conf][nx][ny][nz][1] = rot_y
                    rot_all[conf][nx][ny][nz][2] = rot_z
                    rot_all[conf][nx][ny][nz][3] = rot_w
                
                elif rotation == 'euler_angle':
                    rot = Rotation.from_quat((rot_x, rot_y, rot_z, rot_w))

                    euler_sequence_string={'intrinsic': 'XYZ', 'extrinsic': 'xyz'}
                    rot_euler = rot.as_euler(seq=euler_sequence_string[euler_type])
                    rot_euler = np.asarray(rot_euler)

                    rot_all[conf][nx][ny][nz][0] = rot_euler[0]
                    rot_all[conf][nx][ny][nz][1] = rot_euler[1]
                    rot_all[conf][nx][ny][nz][2] = rot_euler[2]

            else:
                rot_all[conf][nx][ny][nz][0] = 0
                rot_all[conf][nx][ny][nz][1] = 0
                rot_all[conf][nx][ny][nz][2] = 0
                if rotation=='quaternion':
                    rot_all[conf][nx][ny][nz][3] = 0

            
        if lattice_type != 'BCC':
            grain_all[conf][nx][ny][nz] = grain_all[conf][nx][ny][nz] + 1
            
            
np.save(f'{rotation_fname}_vanilla_boundary_{set_}_set_{res}.npy', rot_all)
np.save(f'grain_boundary_vanilla_boundary_{set_}_set_{res}.npy', grain_all)
