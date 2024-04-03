import os
import sys
import numpy as np
import h5py
import collections

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


Examples = collections.namedtuple("Examples", "names, pointsets_A, pointsets_B, pointsets_meta_A, pointsets_T_A, pointsets_R_A, pointsets_T_B, pointsets_R_B")


def shuffle_examples( data ):

    idx = np.arange(  data.names.shape[0] )
    np.random.shuffle(idx)

    return Examples(
        names=data.names[idx, ...],
        pointsets_A=data.pointsets_A[idx, ...],
        pointsets_B=data.pointsets_B[idx, ...],
        pointsets_meta_A=data.pointsets_meta_A[idx, ...],
        pointsets_T_A=data.pointsets_T_A[idx,...],
        pointsets_R_A=data.pointsets_R_A[idx,...],
        pointsets_T_B=data.pointsets_T_B[idx,...],
        pointsets_R_B=data.pointsets_R_B[idx,...],
        # pointsets_T_A=data.pointsets_T_A[idx, 7:24, ...],
        # pointsets_R_A=data.pointsets_R_A[idx, 7:24, ...],
        # pointsets_T_B=data.pointsets_T_B[idx, 7:24, ...],
        # pointsets_R_B=data.pointsets_R_B[idx, 7:24, ...],
    )

def load_examples(h5_filename,  fieldname_A, fieldname_B, meta_A, pre_T, pre_R, post_T, post_R, fieldname_modelname ):

    f = h5py.File(h5_filename)
    pointsets_A = f[fieldname_A][:,:,:]
    pointsets_B = f[fieldname_B][:,:,:]
    pointsets_meta_A = f[meta_A][:,:,:]
    # pointsets_T_A = f[pre_T][:,6:23,:]
    # pointsets_R_A = f[pre_R][:,6:23,:]
    # pointsets_T_B = f[post_T][:,6:23,:]
    # pointsets_R_B = f[post_R][:,6:23,:]
    pointsets_T_A = f[pre_T][:,:,:]
    pointsets_R_A = f[pre_R][:,:,:]
    pointsets_T_B = f[post_T][:,:,:]
    pointsets_R_B = f[post_R][:,:,:]
    names = f[fieldname_modelname][:]
    return Examples(
        names=names,
        pointsets_A=pointsets_A,
        pointsets_B=pointsets_B,
        pointsets_meta_A= pointsets_meta_A,
        pointsets_T_A=pointsets_T_A,
        pointsets_R_A=pointsets_R_A,
        pointsets_T_B=pointsets_T_B,
        pointsets_R_B=pointsets_R_B,
    )
#### for spine with 100 data
def load_eval_examples(h5_filename,  fieldname_A, fieldname_B, meta_A, pre_T, pre_R, post_T, post_R, fieldname_modelname ):

    f = h5py.File(h5_filename)
    pointsets_A = f[fieldname_A][10:20,:,:]
    pointsets_B = f[fieldname_B][10:20,:,:]
    pointsets_meta_A = f[meta_A][10:20,:,:]
    # pointsets_T_A = f[pre_T][0:10,6:23,:]
    # pointsets_R_A = f[pre_R][0:10:,6:23,:]
    # pointsets_T_B = f[post_T][0:10:,6:23,:]
    # pointsets_R_B = f[post_R][0:10:,6:23,:]
    pointsets_T_A = f[pre_T][10:20,:,:]
    pointsets_R_A = f[pre_R][10:20,:,:]
    pointsets_T_B = f[post_T][10:20,:,:]
    pointsets_R_B = f[post_R][10:20,:,:]
    names = f[fieldname_modelname][10:20]
    return Examples(
        names=names,
        pointsets_A=pointsets_A,
        pointsets_B=pointsets_B,
        pointsets_meta_A= pointsets_meta_A,
        pointsets_T_A=pointsets_T_A,
        pointsets_R_A=pointsets_R_A,
        pointsets_T_B=pointsets_T_B,
        pointsets_R_B=pointsets_R_B,
    )


def load_test_examples(h5_filename,  fieldname_A, fieldname_B, meta_A, pre_T, pre_R, post_T, post_R, fieldname_modelname ):

    f = h5py.File(h5_filename)
    pointsets_A = f[fieldname_A][10:20,:,:]
    pointsets_B = f[fieldname_B][10:20,:,:]
    pointsets_meta_A = f[meta_A][10:20,:,:]
    # pointsets_T_A = f[pre_T][10:20,6:23,:]
    # pointsets_R_A = f[pre_R][10:20:,6:23,:]
    # pointsets_T_B = f[post_T][10:20:,6:23,:]
    # pointsets_R_B = f[post_R][10:20:,6:23,:]
    pointsets_T_A = f[pre_T][10:20,:,:]
    pointsets_R_A = f[pre_R][10:20,:,:]
    pointsets_T_B = f[post_T][10:20,:,:]
    pointsets_R_B = f[post_R][10:20,:,:]
    names = f[fieldname_modelname][10:20]
    return Examples(
        names=names,
        pointsets_A=pointsets_A,
        pointsets_B=pointsets_B,
        pointsets_meta_A= pointsets_meta_A,
        pointsets_T_A=pointsets_T_A,
        pointsets_R_A=pointsets_R_A,
        pointsets_T_B=pointsets_T_B,
        pointsets_R_B=pointsets_R_B,
    )

#
# def load_test_examples(h5_filename,  fieldname_A, fieldname_B, meta_A, pre_T, pre_R, post_T, post_R, fieldname_modelname ):
#
#     f = h5py.File(h5_filename)
#     pointsets_A = f[fieldname_A][10:20,:,:]
#     pointsets_B = f[fieldname_B][10:20,:,:]
#     pointsets_meta_A = f[meta_A][10:20,:,:]
#     # pointsets_T_A = f[pre_T][10:20,6:23,:]
#     # pointsets_R_A = f[pre_R][10:20:,6:23,:]
#     # pointsets_T_B = f[post_T][10:20:,6:23,:]
#     # pointsets_R_B = f[post_R][10:20:,6:23,:]
#     pointsets_T_A = f[pre_T][10:20,:,:]
#     pointsets_R_A = f[pre_R][10:20,:,:]
#     pointsets_T_B = f[post_T][10:20,:,:]
#     pointsets_R_B = f[post_R][10:20,:,:]
#     names = f[fieldname_modelname][10:20]
#     return Examples(
#         names=names,
#         pointsets_A=pointsets_A,
#         pointsets_B=pointsets_B,
#         pointsets_meta_A= pointsets_meta_A,
#         pointsets_T_A=pointsets_T_A,
#         pointsets_R_A=pointsets_R_A,
#         pointsets_T_B=pointsets_T_B,
#         pointsets_R_B=pointsets_R_B,
#     )

################################################################
 # f = h5py.File(h5_filename)
 #    pointsets_A = f[fieldname_A][0:2]
 #    pointsets_B = f[fieldname_B][0:2]
 #    names = f[fieldname_modelname][0:2]
 #    meta_data = np.ndarray((2,4),np.float32)
 #    meta_rep = np.zeros((pointsets_A.shape[0], pointsets_A.shape[1], meta_data.shape[1]))
 #    for i in range(meta_data.shape[0]):
 #        meta_rep[i,:,:] = np.repeat(meta_data[i:i+1,:], pointsets_A.shape[1], axis=0)
 #        #meta_rep[i,:,:] = np.expand_dims(meta_rep, axis=0)
 #    data_A = np.concatenate((pointsets_A,meta_rep), axis = 2)
 #    data_B = np.concatenate((pointsets_B, meta_rep), axis=2)
 #    return Examples(
 #        names=names,
 #        #pointsets_A=pointsets_A,
 #        #pointsets_B=pointsets_B,
 #        pointsets_A=data_A,
 #        pointsets_B=data_B,
################################################################

# ### for trunk with 41 data
# def load_eval_examples(h5_filename,  fieldname_A, fieldname_B, meta_A, pre_T, pre_R, post_T, post_R, fieldname_modelname ):
#
#     f = h5py.File(h5_filename)
#     pointsets_A = f[fieldname_A][0:3,:,:]
#     pointsets_B = f[fieldname_B][0:3,:,:]
#     pointsets_meta_A = f[meta_A][0:3,:,:]
#     # pointsets_T_A = f[pre_T][0:3,6:23,:]
#     # pointsets_R_A = f[pre_R][0:3:,6:23,:]
#     # pointsets_T_B = f[post_T][0:3:,6:23,:]
#     # pointsets_R_B = f[post_R][0:3:,6:23,:]
#     pointsets_T_A = f[pre_T][0:3,:,:]
#     pointsets_R_A = f[pre_R][0:3,:,:]
#     pointsets_T_B = f[post_T][0:3,:,:]
#     pointsets_R_B = f[post_R][0:3,:,:]
#     names = f[fieldname_modelname][0:3]
#     return Examples(
#         names=names,
#         pointsets_A=pointsets_A,
#         pointsets_B=pointsets_B,
#         pointsets_meta_A= pointsets_meta_A,
#         pointsets_T_A=pointsets_T_A,
#         pointsets_R_A=pointsets_R_A,
#         pointsets_T_B=pointsets_T_B,
#         pointsets_R_B=pointsets_R_B,
#     )
#
# def load_test_examples(h5_filename,  fieldname_A, fieldname_B, meta_A, pre_T, pre_R, post_T, post_R, fieldname_modelname ):
#
#     f = h5py.File(h5_filename)
#     pointsets_A = f[fieldname_A][0:3,:,:]
#     pointsets_B = f[fieldname_B][0:3,:,:]
#     pointsets_meta_A = f[meta_A][0:3,:,:]
#     # pointsets_T_A = f[pre_T][0:3,6:23,:]
#     # pointsets_R_A = f[pre_R][0:3:,6:23,:]
#     # pointsets_T_B = f[post_T][0:3:,6:23,:]
#     # pointsets_R_B = f[post_R][0:3:,6:23,:]
#     pointsets_T_A = f[pre_T][0:3,:,:]
#     pointsets_R_A = f[pre_R][0:3,:,:]
#     pointsets_T_B = f[post_T][0:3,:,:]
#     pointsets_R_B = f[post_R][0:3,:,:]
#     names = f[fieldname_modelname][0:3]
#     return Examples(
#         names=names,
#         pointsets_A=pointsets_A,
#         pointsets_B=pointsets_B,
#         pointsets_meta_A= pointsets_meta_A,
#         pointsets_T_A=pointsets_T_A,
#         pointsets_R_A=pointsets_R_A,
#         pointsets_T_B=pointsets_T_B,
#         pointsets_R_B=pointsets_R_B,
#     )

###
def output_point_cloud_ply(xyzs, names, output_dir, foldername ):

    if not os.path.exists( output_dir ):
        os.mkdir(  output_dir  )

    plydir = output_dir + '/' + foldername

    if not os.path.exists( plydir ):
        os.mkdir( plydir )

    numFiles = len(names)

    for fid in range(numFiles):

        print('write: ' + plydir +'/'+names[fid]+'.ply')

        with open( plydir +'/'+names[fid]+'.ply', 'w') as f:
            if len(xyzs.shape)==3:
                pn = xyzs.shape[1]
                f.write('ply\n')
                f.write('format ascii 1.0\n')
                f.write('element vertex %d\n' % (pn))
                f.write('property float x\n')
                f.write('property float y\n')
                f.write('property float z\n')
                f.write('end_header\n')
                for i in range(pn):
                   f.write('%f %f %f\n' % (xyzs[fid][i][0],  xyzs[fid][i][1],  xyzs[fid][i][2]) )
                    #f.write('%f %f %f\n' % (xyzs[fid][i][0], xyzs[fid][i][1], xyzs[fid][i][2]))
            else:
                pn = xyzs.shape[0]
                f.write('ply\n')
                f.write('format ascii 1.0\n')
                f.write('element vertex %d\n' % (pn))
                f.write('property float x\n')
                f.write('property float y\n')
                f.write('property float z\n')
                f.write('end_header\n')
                for i in range(pn):
                    f.write('%f %f %f\n' % (xyzs[i][0], xyzs[i][1], xyzs[i][2]))