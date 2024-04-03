from os import listdir
from os.path import isfile, join
import argparse
import h5py
import numpy as np

from plyfile import PlyData

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default='test' )

# pointNum = 2048
pointNum = 3000 #6000 #60000 #108 #131 #35000 #2042 #31055 #48
pointNum_meta= 3000 #6000 #60000 #35000 #108 #48  #131#24#102#131
pointNum_spine = 108
vertebraNum_T=102 #102 #24#16
vertebraNum_R=24 #102 #24#23#16

FLAGS = parser.parse_args()

#sampFolder = FLAGS.modescanFolder = FLAGS.mode + '_scan'
sampFolder = FLAGS.modescanFolder = FLAGS.mode + '_Spine_small'
#sampFolder = FLAGS.modescanFolder = FLAGS.mode + '_Spine_transform'


#scanfiles = [f for f in listdir(scanFolder) if isfile(join(scanFolder, f)) and f.endswith('.ply')  ]
scanfiles = [f for f in listdir(sampFolder) if isfile(join(sampFolder, f)) and f.endswith('.ply')  ]


#numFiles = len(scanfiles)
numFiles = 40 #4 #40 #20 #80


f = h5py.File( sampFolder + ".hdf5", "w")

string_dt = h5py.special_dtype(vlen=str)

names = f.create_dataset("names", (numFiles,), dtype=string_dt)
surface_points = f.create_dataset("surface", (numFiles, pointNum, 3), dtype='f')
skeleton_points = f.create_dataset("skeleton", (numFiles, pointNum, 3), dtype='f')
meta_points = f.create_dataset("meta", (numFiles, pointNum_meta, 6), dtype='f')

dist_points = f.create_dataset("dist", (numFiles, pointNum, 3), dtype='f')
index_points = f.create_dataset("index", (numFiles, pointNum, 2), dtype='f')

spinePre_points = f.create_dataset("spinePre", (numFiles, pointNum_spine, 3), dtype='f')
spinePost_points = f.create_dataset("spinePost", (numFiles, pointNum_spine, 3), dtype='f')


# pre_T = f.create_dataset("pre_T", (numFiles, vertebraNum_T, 3), dtype='f')
# pre_R = f.create_dataset("pre_R", (numFiles, vertebraNum_R, 3), dtype='f')
# post_T = f.create_dataset("post_T", (numFiles, vertebraNum_T, 3), dtype='f')
# post_R = f.create_dataset("post_R", (numFiles, vertebraNum_R, 3), dtype='f')


gid=0
for fid in range(numFiles):

    scanfiles.sort()
    scanfname = scanfiles[gid]
    basename1 = scanfname[0:8]   #Trunk: scanfname[0:8]  #spine:scanfname[0:10]    #basename1 = scanfname[0:10]
    basename2 = 'ascend_' #'ascend_' #  basename2=''
    basename3 = '' #'LCN' #LCN MSN
    ####scanfname = scanfiles[fid]

    #### skel_path  = sampFolder + '/' + basename1 + 'pre_new_xy.ply'
    #### surf_path  = sampFolder + '/' + basename1 +'post_new_xy.ply'

    # skel_path = sampFolder + '/' + basename1 + 'pre' + '.ply'
    # surf_path = sampFolder + '/' + basename1 + 'post' + '.ply'
    # # skel_path = sampFolder + '/' + basename1 + 'pre_normalized_'+ basename3 + '.ply'
    # # surf_path = sampFolder + '/' + basename1 + 'post_normalized_'+ basename3 + '.ply'
    # meta_path = sampFolder + '/' + basename1 + 'meta_landmarks.ply'
    #
    skel_path = sampFolder + '/' + basename1 + 'pre_trunk.ply'
    surf_path = sampFolder + '/' + basename1 + 'post_trunk.ply'
    meta_path = sampFolder + '/' + basename1 + 'meta.ply'

    symdist_path = sampFolder + '/' + basename1 + 'post_symdist.ply'
    symindex_path = sampFolder + '/' + basename1 + 'post_symindex.ply'

    spinePre_path = sampFolder + '/' + basename1 + 'pre_spine.ply'
    spinePost_path = sampFolder + '/' + basename1 + 'post_spine.ply'
    ################
    # pre_T_path = sampFolder + '/' + basename1 + basename2 + 'pre_point_T.ply'
    # pre_R_path = sampFolder + '/' + basename1 + basename2 + 'pre_center_R.ply'
    # post_T_path = sampFolder + '/' + basename1 + basename2 + 'post_point_T.ply'
    # post_R_path = sampFolder + '/' + basename1 + basename2 + 'post_center_R.ply'

    # meta_path = sampFolder + '/' + basename1 + 'meta_T.ply'
    # pre_T_path = sampFolder + '/' + basename1 + basename2 + 'pre_center_normalized_' + basename3 + '_transfo_T.ply'
    # pre_R_path = sampFolder + '/' + basename1 + basename2 + 'pre_center_normalized_' + basename3 + '_transfo_R.ply'
    # post_T_path = sampFolder + '/' + basename1 + basename2 + 'post_center_normalized_' + basename3 + '_transfo_T.ply'
    # post_R_path = sampFolder + '/' + basename1 + basename2 + 'post_center_normalized_' + basename3 + '_transfo_R.ply'

    # pre_T_path = sampFolder + '/' + basename1 + basename2 + 'pre_center_without_normalized' + basename3 + '_transfo_T.ply'
    # pre_R_path = sampFolder + '/' + basename1 + basename2 + 'pre_center_without_normalized' + basename3 + '_transfo_R.ply'
    # post_T_path = sampFolder + '/' + basename1 + basename2 + 'post_center_without_normalized' + basename3 + '_transfo_T.ply'
    # post_R_path = sampFolder + '/' + basename1 + basename2 + 'post_center_without_normalized' + basename3 + '_transfo_R.ply'

    # pre_T_path = sampFolder + '/' + basename1 + basename2 + 'pre_point' + basename3 + '_T.ply'
    # pre_R_path = sampFolder + '/' + basename1 + basename2 + 'pre_center_without_normalized_transfo' + basename3 + '_R.ply'
    # post_T_path = sampFolder + '/' + basename1 + basename2 + 'post_point' + basename3 + '_T.ply'
    # post_R_path = sampFolder + '/' + basename1 + basename2 + 'post_center_without_normalized_transfo' + basename3 + '_R.ply'
    ### skel_path = sampFolder + '/' + 'Spine.2171510.pre.ply'
    ### surf_path = sampFolder + '/' + 'Spine.2171510.post.ply'
    #### scan_path  = scanFolder + '/' + scanfname
    ################
    skelPlyData = PlyData.read( skel_path )
    surfPlyData = PlyData.read( surf_path )
    metaPlyData = PlyData.read( meta_path )

    distPlyData = PlyData.read(symdist_path)
    indexPlyData = PlyData.read(symindex_path)

    spinePrePlyData = PlyData.read(spinePre_path)
    spinePostPlyData = PlyData.read(spinePost_path)

    # preTPlyData = PlyData.read( pre_T_path )
    # preRPlyData = PlyData.read( pre_R_path)
    # postTPlyData = PlyData.read(post_T_path)
    # postRPlyData = PlyData.read( post_R_path)

    names[fid] = scanfname[0:7]
    #### names[ fid] = scanfname[0:-6]
    #### names[fid] = 'Spine'
    ################
    skeleton_points[ fid, :, 0 ] = skelPlyData['vertex']['x']
    skeleton_points[ fid, :, 1 ] = skelPlyData['vertex']['y']
    skeleton_points[ fid, :, 2 ] = skelPlyData['vertex']['z']

    surface_points[ fid, :, 0 ] = surfPlyData['vertex']['x']
    surface_points[ fid, :, 1 ] = surfPlyData['vertex']['y']
    surface_points[ fid, :, 2 ] = surfPlyData['vertex']['z']

    meta_points[fid, :, 0] = metaPlyData['vertex']['sex']
    meta_points[ fid, :, 1 ] = metaPlyData['vertex']['age']
    meta_points[ fid, :, 2 ] = metaPlyData['vertex']['height']
    meta_points[ fid, :, 3 ] = metaPlyData['vertex']['weight']
    meta_points[ fid, :, 4 ] = metaPlyData['vertex']['MBI']
    meta_points[fid, :, 5] = metaPlyData['vertex']['surgeon']

    dist_points[fid, :, 0] = distPlyData['vertex']['x']
    dist_points[fid, :, 1] = distPlyData['vertex']['y']
    dist_points[fid, :, 2] = distPlyData['vertex']['z']
    # # #
    index_points[fid, :, 0] = indexPlyData['vertex']['index1']
    index_points[fid, :, 1] = indexPlyData['vertex']['index2']

    spinePre_points[fid, :, 0] = spinePrePlyData['vertex']['x']
    spinePre_points[fid, :, 1] = spinePrePlyData['vertex']['y']
    spinePre_points[fid, :, 2] = spinePrePlyData['vertex']['z']

    spinePost_points[fid, :, 0] = spinePostPlyData['vertex']['x']
    spinePost_points[fid, :, 1] = spinePostPlyData['vertex']['y']
    spinePost_points[fid, :, 2] = spinePostPlyData['vertex']['z']
    ################
    # pre_T[fid, :, 0] = preTPlyData['vertex']['x']
    # pre_T[fid, :, 1] = preTPlyData['vertex']['y']
    # pre_T[fid, :, 2] = preTPlyData['vertex']['z']
    #
    # pre_R[fid, :, 0] = preRPlyData['vertex']['x']
    # pre_R[fid, :, 1] = preRPlyData['vertex']['y']
    # pre_R[fid, :, 2] = preRPlyData['vertex']['z']
    #
    # post_T[fid, :, 0] = postTPlyData['vertex']['x']
    # post_T[fid, :, 1] = postTPlyData['vertex']['y']
    # post_T[fid, :, 2] = postTPlyData['vertex']['z']
    # #
    # post_R[fid, :, 0] = postRPlyData['vertex']['x']
    # post_R[fid, :, 1] = postRPlyData['vertex']['y']
    # post_R[fid, :, 2] = postRPlyData['vertex']['z']

    ################
    # pre_T[fid, :, 0] = preTPlyData['vertex']['x']
    # pre_T[fid, :, 1] = preTPlyData['vertex']['y']
    # pre_T[fid, :, 2] = preTPlyData['vertex']['z']
    #
    # pre_R[fid, :, 0] = preRPlyData['vertex']['x']
    # pre_R[fid, :, 1] = preRPlyData['vertex']['y']
    # pre_R[fid, :, 2] = preRPlyData['vertex']['z']

    #### pre_R[fid, :, 0, 0] = preRPlyData['vertex']['x_x']
    #### pre_R[fid, :, 0, 1] = preRPlyData['vertex']['x_y']
    #### pre_R[fid, :, 0, 2] = preRPlyData['vertex']['x_z']
    #### pre_R[fid, :, 1, 0] = preRPlyData['vertex']['y_x']
    #### pre_R[fid, :, 1, 1] = preRPlyData['vertex']['y_y']
    #### pre_R[fid, :, 1, 2] = preRPlyData['vertex']['y_z']
    #### pre_R[fid, :, 2, 0] = preRPlyData['vertex']['z_x']
    #### pre_R[fid, :, 2, 1] = preRPlyData['vertex']['z_y']
    #### pre_R[fid, :, 2, 2] = preRPlyData['vertex']['z_z']

    # post_T[fid, :, 0] = postTPlyData['vertex']['x']
    # post_T[fid, :, 1] = postTPlyData['vertex']['y']
    # post_T[fid, :, 2] = postTPlyData['vertex']['z']
    #
    # post_R[fid, :, 0] = postRPlyData['vertex']['x']
    # post_R[fid, :, 1] = postRPlyData['vertex']['y']
    # post_R[fid, :, 2] = postRPlyData['vertex']['z']

    #### post_R[fid, :, 0, 0] = postRPlyData['vertex']['x_x']
    #### post_R[fid, :, 0, 1] = postRPlyData['vertex']['x_y']
    #### post_R[fid, :, 0, 2] = postRPlyData['vertex']['x_z']
    #### post_R[fid, :, 1, 0] = postRPlyData['vertex']['y_x']
    #### post_R[fid, :, 1, 1] = postRPlyData['vertex']['y_y']
    #### post_R[fid, :, 1, 2] = postRPlyData['vertex']['y_z']
    #### post_R[fid, :, 2, 0] = postRPlyData['vertex']['z_x']
    #### post_R[fid, :, 2, 1] = postRPlyData['vertex']['z_y']
    #### post_R[fid, :, 2, 2] = postRPlyData['vertex']['z_z']

    #### scan_points[ fid, :, 0 ] = scanPlyData['vertex']['x']
    #### scan_points[ fid, :, 1 ] = scanPlyData['vertex']['y']
    #### scan_points[ fid, :, 2 ] = scanPlyData['vertex']['z']
    gid= gid + 7 #3 #5 #7

names.flush()
skeleton_points.flush()
surface_points.flush()
meta_points.flush()
# spinePre_points.flush()
# spinePost_points.flush()
dist_points.flush()
index_points.flush()
# pre_T.flush()
# pre_R.flush()
# post_T.flush()
# post_R.flush()
#### scan_points.flush()
