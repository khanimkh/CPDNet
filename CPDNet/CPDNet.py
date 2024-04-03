import os
import sys
import collections
from numpy import linalg as LA
import cv2
from tensorflow.python.keras import backend
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow import linalg as _linalg
#from tensorflow.python.ops.linalg import linalg_impl as _linalg



BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(   BASE_DIR + "/pointnet_plusplus/utils")
sys.path.append(   BASE_DIR + "/pointnet_plusplus/tf_ops")
sys.path.append(   BASE_DIR + "/pointnet_plusplus/tf_ops/3d_interpolation")
sys.path.append(   BASE_DIR + "/pointnet_plusplus/tf_ops/grouping")
sys.path.append(   BASE_DIR + "/pointnet_plusplus/tf_ops/sampling")
import tensorflow as tf
import numpy as np
import ioUtil
#import tensorflow_probability as tfp
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_fp_module
import autograd.numpy as np
from autograd import grad
from autograd import elementwise_grad as egrad
np.set_printoptions(precision=4)


Model = collections.namedtuple("Model", \
                               "pointSet_A_ph,  pointSet_B_ph, pointSet_meta_A_ph, pointSet_T_A_ph, pointSet_R_A_ph, pointSet_T_B_ph, pointSet_R_B_ph\
                               is_training_ph,\
                               Predicted_A, Predicted_B, displace_BA, displace_AB, list_dis_A2B_all, list_dis_B2A_all, Accuracy\
                               data_loss_A, shapeLoss_A, densityLoss_A, \
                               data_loss_B, shapeLoss_B, densityLoss_B, \
                               total_loss, \
                               rigid_loss_A, rigid_loss_B\
                               regul_loss, \
                               data_train, rigid_train, total_train, \
                               learning_rate,  global_step,  bn_decay, \
                               training_sum_ops, testing_sum_ops,\
                               train_dataloss_A_ph,  train_dataloss_B_ph, train_totalloss_ph,  train_regul_ph, train_rigidloss_A_ph, train_rigidloss_B_ph\
                               test_dataloss_A_ph,   test_dataloss_B_ph, test_totalloss_ph, test_regul_ph, test_rigidloss_A_ph, test_rigidloss_B_ph"                    )

def create_model( FLAGS  ):

    ############################################################
    ####################  Hyper-parameters   ####################
    ##############################################################

    global_step = tf.train.get_or_create_global_step()

    learning_rate = tf.train.exponential_decay(
        FLAGS.learning_rate,  # base learning rate 0.001
        global_step   * FLAGS.batch_size,  # global_var indicating the number of steps
        FLAGS.example_num  * FLAGS.decayEpoch,  # step size
        0.5,  # decay rate
        staircase=True
    )
    learning_rate = tf.maximum(learning_rate, 1e-4)

    bn_momentum = tf.train.exponential_decay(
        0.001, # 0.001
        global_step  * FLAGS.batch_size,  # global_var indicating the number of steps
        FLAGS.example_num * FLAGS.decayEpoch * 2,     # step size,
        0.5,   # decay rate
        staircase=True
    )
    bn_decay = tf.minimum(0.99,   1 - bn_momentum)

    Train_examples = ioUtil.load_examples(FLAGS.train_hdf5, FLAGS.domain_A, FLAGS.domain_B, FLAGS.meta_A, FLAGS.pre_T, FLAGS.pre_R, FLAGS.post_T, FLAGS.post_R, 'names')
    Test_examples = ioUtil.load_examples(FLAGS.test_hdf5, FLAGS.domain_A, FLAGS.domain_B, FLAGS.meta_A, FLAGS.pre_T, FLAGS.pre_R, FLAGS.post_T, FLAGS.post_R, 'names')

    ##############################################################
    ####################  Create the network  ####################
    ##############################################################

    # if FLAGS.mode == 'train':
    #     pointSet_A_ph = tf.placeholder_with_default(Train_examples.pointsets_A, shape=(FLAGS.batch_size, FLAGS.point_num, 3))
    #     pointSet_B_ph = tf.placeholder_with_default(Train_examples.pointsets_B, shape=(FLAGS.batch_size, FLAGS.point_num, 3))
    #     # Train_examples_shuffled = ioUtil.shuffle_examples(Train_examples)
    #     pointSet_meta_A_ph = tf.placeholder_with_default(Train_examples.pointsets_meta_A, shape=(FLAGS.batch_size, FLAGS.point_num, 6))
    #     pointSet_T_A_ph = tf.placeholder_with_default(Train_examples.pointsets_T_A, shape=(FLAGS.batch_size, FLAGS.vertebra_num, 3))
    #     pointSet_R_A_ph = tf.placeholder_with_default(Train_examples.pointsets_R_A, shape=(FLAGS.batch_size, 24, 3))
    #     pointSet_T_B_ph = tf.placeholder_with_default(Train_examples.pointsets_T_B, shape=(FLAGS.batch_size, FLAGS.vertebra_num, 3))
    #     pointSet_R_B_ph = tf.placeholder_with_default(Train_examples.pointsets_R_B, shape=(FLAGS.batch_size, 24, 3))
    # else:
    #     pointSet_A_ph = tf.placeholder_with_default(Test_examples.pointsets_A, shape=(FLAGS.batch_size, FLAGS.point_num, 3))
    #     pointSet_B_ph = tf.placeholder_with_default(Test_examples.pointsets_B, shape=(FLAGS.batch_size, FLAGS.point_num, 3))
    #     # Test_examples_shuffled = ioUtil.shuffle_examples(Test_examples)
    #     pointSet_meta_A_ph = tf.placeholder_with_default(Train_examples.pointsets_meta_A, shape=(FLAGS.batch_size, FLAGS.point_num, 6))
    #     pointSet_T_A_ph = tf.placeholder_with_default(Test_examples.pointsets_T_A, shape=(FLAGS.batch_size, FLAGS.vertebra_num, 3))
    #     pointSet_R_A_ph = tf.placeholder_with_default(Test_examples.pointsets_R_A, shape=(FLAGS.batch_size, 24, 3))
    #     pointSet_T_B_ph = tf.placeholder_with_default(Test_examples.pointsets_T_B, shape=(FLAGS.batch_size, FLAGS.vertebra_num, 3))
    #     pointSet_R_B_ph = tf.placeholder_with_default(Test_examples.pointsets_R_B, shape=(FLAGS.batch_size, 24, 3))

    pointSet_A_ph = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.point_num, 3) )
    pointSet_B_ph = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.point_num, 3) )
    pointSet_meta_A_ph = tf.placeholder( tf.float32, shape=(FLAGS.batch_size, FLAGS.point_num_meta, 6) )
    pointSet_T_A_ph = tf.placeholder( tf.float32, shape=(FLAGS.batch_size, FLAGS.vertebra_num, 3) )
    #pointSet_R_A_ph = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.vertebra_num, 3, 3))
    #pointSet_R_A_ph = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.vertebra_num, 3))
    pointSet_R_A_ph = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, 24, 3))
    pointSet_T_B_ph = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.vertebra_num, 3))
    #pointSet_R_B_ph = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.vertebra_num, 3 , 3))
    #pointSet_R_B_ph = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.vertebra_num, 3))
    pointSet_R_B_ph = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, 24, 3))

    Predicted_A = tf.placeholder( tf.float32, shape=(FLAGS.batch_size, FLAGS.point_num, 3) )
    Predicted_B = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.point_num, 3))

    displace_A2B = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.point_num, 3))
    displace_B2A = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.point_num, 3))

    dis_A2B = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.point_num, 3))
    dis_B2A = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.point_num, 3))
    Accuracy= tf.constant(0.0, dtype=tf.float32)

    noise_vertebra = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.point_num, 3))

    # pointSet_A_ph=Train_examples.pointsets_A
    # pointSet_B_ph=Train_examples.pointsets_B
    # Train_examples_shuffled = ioUtil.shuffle_examples(Train_examples)
    # pointSet_meta_A_ph = Train_examples_shuffled.pointsets_meta_A
    # pointSet_T_A_ph = Train_examples.pointsets_T_A
    # pointSet_R_A_ph = Train_examples.pointsets_R_A
    # pointSet_T_B_ph = Train_examples.pointsets_T_B
    # pointSet_R_B_ph = Train_examples.pointsets_R_B
    # pointSet_A_ph = tf.placeholder( tf.float32, shape=(FLAGS.batch_size, FLAGS.point_num, 6) )
    # pointSet_B_ph = tf.placeholder( tf.float32, shape=(FLAGS.batch_size, FLAGS.point_num, 6) )
    is_training_ph = tf.placeholder( tf.bool, shape=() )
    # is_training_ph = tf.placeholder_with_default(True,())

    noise1 = None
    noise2 = None
    if FLAGS.noiseLength > 0:
        noise1 = tf.random_normal(shape=[FLAGS.batch_size, FLAGS.point_num, FLAGS.noiseLength], mean=0.0, stddev=1, dtype=tf.float32)
        noise2 = tf.random_normal(shape=[FLAGS.batch_size, FLAGS.point_num, FLAGS.noiseLength], mean=0.0, stddev=1, dtype=tf.float32)

        # list_noise_vertebra = []
        # noise_vertebra = tf.zeros((FLAGS.batch_size, FLAGS.point_num, 3), dtype=np.float32)
        # noise_vertebra_all = tf.zeros((FLAGS.batch_size, FLAGS.point_num, 3), dtype=np.float32)
        #
        # for j in range(0, FLAGS.batch_size, 1):
        #     slice1_ = tf.slice(noise1, [j, 0, 0], [1, 1, -1])
        #     # slice1 = displace_A2B[j,0,:]
        #     slice1 = tf.tile(slice1_, [1, 6, 1])
        #     noise_vertebra = slice1
        #     for i in range(1, 17, 1):
        #         slice2_ = tf.slice(noise1, [j, i, 0], [1, 1, -1])
        #         # slice2 = displace_A2B[j, i, :]
        #         slice2 = tf.tile(slice2_, [1, 6, 1])
        #         noise_vertebra = tf.concat([noise_vertebra, slice2], 1)
        #     list_noise_vertebra.append(noise_vertebra)
        #
        # noise_vertebra = tf.stack(list_noise_vertebra)
        # noise_vertebra_all = tf.squeeze(noise_vertebra)

                # pointSet_A_ph = np.array(pointSet_A_ph)
    # row=len(pointSet_A_ph)
    # if row == 102:
    #     x = (pointSet_A_ph[0, 101, 0] + pointSet_A_ph[0, 100, 0]) / 2
    #     y = (pointSet_A_ph[0, 101, 1] + pointSet_A_ph[0, 100, 1]) / 2
    #     z = (pointSet_A_ph[0, 101, 2] + pointSet_A_ph[0, 100, 2]) / 2
    #     ref = np.array([x, y, z])
    #     c = np.zeros((17, 3),dtype=np.float32)
    #     j = 0
    #     for i in range(0, 101, 6):
    #         # center
    #         c[j, 0] = (pointSet_A_ph[0, i + 5, 0] + pointSet_A_ph[0, i + 4, 0]) / 2 - ref[0]
    #         c[j, 1] = (pointSet_A_ph[0, i + 5, 1] + pointSet_A_ph[0, i + 4, 1]) / 2 - ref[1]
    #         c[j, 2] = (pointSet_A_ph[0, i + 5, 2] + pointSet_A_ph[0, i + 4, 2]) / 2 - ref[2]
    #         j=j+1
    #
    # c = tf.convert_to_tensor(c)
    # c = tf.expand_dims(c, 0)
    list_dis = []
    list_dis_A2B = tf.zeros((FLAGS.batch_size, FLAGS.point_num, 3), dtype=np.float32)
    list_dis_A2B_all = tf.zeros((FLAGS.batch_size, FLAGS.point_num, 3), dtype=np.float32)
    with tf.variable_scope("p2pnet_A2B") as scope:
        displace_A2B = get_displacements_transformation(pointSet_A_ph, is_training_ph, noise1, pointSet_meta_A_ph, FLAGS, bn_decay)
        # displace_A2B = get_displacements( pointSet_A_ph, pointSet_T_A_ph, is_training_ph, noise1, pointSet_meta_A_ph,  FLAGS, bn_decay  )
        # displace_A2B = get_displacements_AE( pointSet_A_ph, is_training_ph, noise1, pointSet_meta_A_ph,  FLAGS, bn_decay  )
        dis = displace_A2B
        # for j in range(0, FLAGS.batch_size, 1):
        #     slice1_ = tf.slice(dis, [j, 0, 0], [1, 1, -1])
        #     # slice1 = displace_A2B[j,0,:]
        #     slice1 = tf.tile(slice1_, [1, 6, 1])
        #     dis_A2B = slice1
        #     for i in range(6, 101, 6): # range(1, 17, 1):
        #         slice2_ = tf.slice(dis, [j, i, 0], [1, 1, -1])
        #         # slice2 = displace_A2B[j, i, :]
        #         slice2 = tf.tile(slice2_, [1, 6, 1])
        #         dis_A2B = tf.concat([dis_A2B, slice2], 1)
        #     list_dis.append(dis_A2B)
        #
        # list_dis_A2B = tf.stack(list_dis)
        # list_dis_A2B_all = tf.squeeze(list_dis_A2B)

    list_dis = []
    list_dis_B2A = tf.zeros((FLAGS.batch_size, FLAGS.point_num, 3), dtype=np.float32)
    list_dis_B2A_all = tf.zeros((FLAGS.batch_size, FLAGS.point_num, 3), dtype=np.float32)
    with tf.variable_scope("p2pnet_B2A") as scope:
        displace_B2A = get_displacements_transformation(pointSet_B_ph, is_training_ph, noise1, pointSet_meta_A_ph, FLAGS, bn_decay)
        # displace_B2A = get_displacements( pointSet_B_ph, pointSet_T_B_ph, is_training_ph, noise2, pointSet_meta_A_ph, FLAGS, bn_decay  )
        # displace_B2A = get_displacements_AE(pointSet_B_ph, is_training_ph, noise1, pointSet_meta_A_ph, FLAGS, bn_decay)
        dis = displace_B2A
    #     for j in range(0, FLAGS.batch_size, 1):
    #         slice1 = tf.slice(dis, [j, 0, 0], [1, 1, -1])
    #         # slice1 = displace_A2B[j,0,:]
    #         slice1 = tf.tile(slice1, [1, 6, 1])
    #         dis_B2A = slice1
    #         for i in range(6, 101, 6): #(1, 17, 1):
    #             slice2 = tf.slice(dis, [j, i, 0], [1, 1, -1])
    #             # slice2 = displace_A2B[j, i, :]
    #             slice2 = tf.tile(slice2, [1, 6, 1])
    #             dis_B2A = tf.concat([dis_B2A, slice2], 1)
    #         list_dis.append(dis_B2A)
    #
    #     list_dis_B2A = tf.stack(list_dis)
    #     list_dis_B2A_all = tf.squeeze(list_dis_B2A)
    #
    # Predicted_A = pointSet_B_ph + list_dis_B2A_all
    # Predicted_B = pointSet_A_ph + list_dis_A2B_all

    # Predicted_A = pointSet_B_ph + dis_B2A * 20
    # Predicted_B = pointSet_A_ph + dis_A2B * 20

    Predicted_A = pointSet_B_ph + displace_B2A
    Predicted_B = pointSet_A_ph + displace_A2B

    # Predicted_A = displace_B2A
    # Predicted_B = displace_A2B

    rigid_loss_A = rigid_transform_3D(Predicted_A,pointSet_T_A_ph,pointSet_R_A_ph,FLAGS) #predict pre
    rigid_loss_A = rigid_loss_A
    #rigid_loss_A=tf.constant(0.0, dtype=tf.float32)
    rigid_loss_B = rigid_transform_3D(Predicted_B,pointSet_T_B_ph,pointSet_R_B_ph,FLAGS) #predict post
    rigid_loss_B = rigid_loss_B
    #rigid_loss_B = tf.constant(0.0, dtype=tf.float32)


    shapeLoss_A, densityLoss_A = get_Geometric_Loss_PointWise(Predicted_A, pointSet_A_ph, FLAGS)
    # shapeLoss_A, densityLoss_A = get_Geometric_Loss(Predicted_A, pointSet_A_ph, FLAGS)
    #data_loss_A, shapeLoss_A, densityLoss_A = get_Geometric_Loss(Predicted_A, pointSet_A_ph, FLAGS)

    # shapeLoss_A=tf.constant(0.0, dtype=tf.float32)
    # data_loss_A=tf.constant(0.0, dtype=tf.float32)
    # densityLoss_A=tf.constant(0.0, dtype=tf.float32)

    Accuracy=get_Accuracy_Rec(Predicted_B, pointSet_B_ph, FLAGS)

    shapeLoss_B, densityLoss_B = get_Geometric_Loss_PointWise(Predicted_B, pointSet_B_ph, FLAGS)
    # shapeLoss_B, densityLoss_B = get_Geometric_Loss(Predicted_B, pointSet_B_ph, FLAGS)
    #data_loss_B, shapeLoss_B, densityLoss_B = get_Geometric_Loss(Predicted_B, pointSet_B_ph, FLAGS)

    if FLAGS.regularWeight > 0:
        regul_loss = get_Regularizing_Loss(pointSet_A_ph, pointSet_B_ph,  Predicted_A, Predicted_B)
    else:
        regul_loss = tf.constant(0.0, dtype=tf.float32)

    # if FLAGS.localWeight > 0:
    #     local_loss = get_Local_Loss(pointSet_A_ph, pointSet_B_ph, Predicted_A)
    # else:
    #     local_loss = tf.constant(0.0, dtype=tf.float32)
    #local_loss = tf.constant(0.0, dtype=tf.float32)

    #RigidLoss =  rigid_loss_A + rigid_loss_B
    RigidLoss = rigid_loss_B

    data_loss_A = shapeLoss_A * FLAGS.shapeWeight + densityLoss_A * FLAGS.densityWeight
    data_loss_B = shapeLoss_B * FLAGS.shapeWeight + densityLoss_B * FLAGS.densityWeight
    #DataLoss = data_loss_A + data_loss_B
    DataLoss = data_loss_B

    TotalLoss = DataLoss
    # TotalLoss = DataLoss +  RigidLoss * FLAGS.localWeight
    #TotalLoss = DataLoss + regul_loss * FLAGS.regularWeight + RigidLoss * FLAGS.localWeight
    #TotalLoss = DataLoss + regul_loss * FLAGS.regularWeight
    ########TotalLoss = DataLoss + regul_loss * FLAGS.regularWeight + local_loss * FLAGS.localWeight

    train_variables = tf.trainable_variables()
    trainer = tf.train.AdamOptimizer(learning_rate)
    ###tf.enable_eager_execution()

    #val_list = lambda: Model.Predicted_A
    #rigid_train_op = trainer.minimize(RigidLoss, var_list=[val_list])
    rigid_train_op = trainer.minimize(RigidLoss, var_list=train_variables, global_step=global_step)
    data_train_op = trainer.minimize(DataLoss, var_list=train_variables, global_step=global_step)
    total_train_op = trainer.minimize(TotalLoss, var_list=train_variables, global_step=global_step)

    rigid_train = rigid_train_op
    data_train  = data_train_op
    total_train = total_train_op

    ##############################################################
    ####################  Create summarizers  ####################
    ##############################################################

    train_totalloss_ph = tf.placeholder(tf.float32, shape=())
    train_dataloss_A_ph = tf.placeholder(tf.float32, shape=())
    train_dataloss_B_ph = tf.placeholder(tf.float32, shape=())
    train_regul_ph = tf.placeholder(tf.float32, shape=())
    train_rigidloss_A_ph = tf.placeholder(tf.float32, shape=())
    train_rigidloss_B_ph = tf.placeholder(tf.float32, shape=())

    test_totalloss_ph = tf.placeholder(tf.float32, shape=())
    test_dataloss_A_ph = tf.placeholder(tf.float32, shape=())
    test_dataloss_B_ph = tf.placeholder(tf.float32, shape=())
    test_regul_ph = tf.placeholder(tf.float32, shape=())
    test_rigidloss_A_ph = tf.placeholder(tf.float32, shape=())
    test_rigidloss_B_ph = tf.placeholder(tf.float32, shape=())


    lr_sum_op = tf.summary.scalar('learning rate', learning_rate)
    global_step_sum_op = tf.summary.scalar('batch_number', global_step)

    train_totalloss_sum_op = tf.summary.scalar('train_totalloss', train_totalloss_ph)
    train_dataloss_A_sum_op = tf.summary.scalar('train_dataloss_A', train_dataloss_A_ph)
    train_dataloss_B_sum_op = tf.summary.scalar('train_dataloss_B', train_dataloss_B_ph)
    train_regul_sum_op = tf.summary.scalar('train_regul', train_regul_ph)
    train_rigidloss_A_sum_op = tf.summary.scalar('train_rigidloss_A', train_rigidloss_A_ph)
    train_rigidloss_B_sum_op = tf.summary.scalar('train_rigidloss_B', train_rigidloss_B_ph)

    test_totalloss_sum_op = tf.summary.scalar('test_totalloss', test_totalloss_ph)
    test_dataloss_A_sum_op = tf.summary.scalar('test_dataloss_A', test_dataloss_A_ph)
    test_dataloss_B_sum_op = tf.summary.scalar('test_dataloss_B', test_dataloss_B_ph)
    test_regul_sum_op = tf.summary.scalar('test_regul', test_regul_ph)
    test_rigidloss_A_sum_op = tf.summary.scalar('test_rigidloss_A', test_rigidloss_A_ph)
    test_rigidloss_B_sum_op = tf.summary.scalar('test_rigidloss_B', test_rigidloss_B_ph)


    training_sum_ops = tf.summary.merge( \
        [lr_sum_op, train_totalloss_sum_op, train_dataloss_A_sum_op, train_dataloss_B_sum_op, train_regul_sum_op, train_rigidloss_A_sum_op, train_rigidloss_B_sum_op])

    testing_sum_ops = tf.summary.merge( \
        [test_totalloss_sum_op, test_dataloss_A_sum_op, test_dataloss_B_sum_op, test_regul_sum_op, test_rigidloss_A_sum_op, test_rigidloss_B_sum_op ])

    return Model(
        pointSet_A_ph=pointSet_A_ph,  pointSet_B_ph=pointSet_B_ph,
        Predicted_A=Predicted_A,Predicted_B=Predicted_B, displace_BA=displace_B2A, displace_AB=displace_A2B, list_dis_A2B_all=list_dis_A2B_all, list_dis_B2A_all=list_dis_B2A_all, Accuracy=Accuracy,
        pointSet_T_A_ph=pointSet_T_A_ph, pointSet_R_A_ph=pointSet_R_A_ph,
        pointSet_T_B_ph=pointSet_T_B_ph, pointSet_R_B_ph=pointSet_R_B_ph,
        pointSet_meta_A_ph=pointSet_meta_A_ph,
        is_training_ph=is_training_ph,
        data_loss_A=data_loss_A,   shapeLoss_A=shapeLoss_A,     densityLoss_A=densityLoss_A,
        data_loss_B=data_loss_B,   shapeLoss_B=shapeLoss_B,     densityLoss_B=densityLoss_B,
        total_loss=TotalLoss,
        rigid_loss_A=rigid_loss_A,
        rigid_loss_B=rigid_loss_B,
        regul_loss=regul_loss,
        data_train=data_train,    rigid_train=rigid_train,   total_train=total_train,
        learning_rate=learning_rate, global_step=global_step, bn_decay=bn_decay,
        training_sum_ops=training_sum_ops, testing_sum_ops=testing_sum_ops,
        train_dataloss_A_ph=train_dataloss_A_ph, train_dataloss_B_ph=train_dataloss_B_ph, train_totalloss_ph=train_totalloss_ph,train_regul_ph=train_regul_ph, train_rigidloss_A_ph=train_rigidloss_A_ph, train_rigidloss_B_ph=train_rigidloss_B_ph,
        test_dataloss_A_ph=test_dataloss_A_ph, test_dataloss_B_ph=test_dataloss_B_ph, test_totalloss_ph=test_totalloss_ph, test_regul_ph=test_regul_ph, test_rigidloss_A_ph=test_rigidloss_A_ph, test_rigidloss_B_ph=test_rigidloss_B_ph
    )

def get_displacements_transformation(input_vertebra, is_training, noise, meta_A, FLAGS, bn_decay=None):
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """

    batch_size = FLAGS.batch_size
    num_points = FLAGS.point_num

    l0_vertebra = input_vertebra
    # depth = 3
    # meta_A_on_hot=tf.one_hot(meta_A, depth, on_value=1.0, off_value=0.0, axis=-1)

    # #
    if noise is not None:
        l0_vertebra = tf.concat(axis=2, values=[l0_vertebra, noise])
    # # # # # # #     # meta_data = np.ndarray((1, 4), np.float32)
    # # # # # # #     # meta_data = tf.zeros(shape=[FLAGS.batch_size, FLAGS.point_num, FLAGS.metaLength], dtype=tf.float32, name=None)
    # # # # # # #
    # if meta_A is not None:
    #     #l0_vertebra = tf.concat(axis=2, values=[l0_vertebra, meta_A * FLAGS.metaWeight, meta_A * FLAGS.metaWeight, meta_A * FLAGS.metaWeight, meta_A * FLAGS.metaWeight])
    #     l0_vertebra = tf.concat(axis=2, values=[l0_vertebra, meta_A])


        # l0_vertebra = tf.concat(axis=2, values=[l0_vertebra, meta_A * FLAGS.metaWeight, meta_A * FLAGS.metaWeight,
        #                                       meta_A * FLAGS.metaWeight, meta_A * FLAGS.metaWeight])
        # l0_points = tf.concat(axis=2, values=[l0_points, meta_A * FLAGS.metaWeight])
        ## l0_points = tf.concat(axis=2, values=[l0_points, meta_A[0:1, 0:17, :]])
        ##l0_points = np.concatenate((l0_points, meta_A), axis=2)


    # Construct MLP model
    # net = multilayer_perceptron(l0_vertebra, FLAGS)
    # displacements = tf.sigmoid(net) * FLAGS.range_max * 2 - FLAGS.range_max

    # Construct Conv model
    # net = tf_util.conv1d(l0_vertebra, 128, 1, padding='VALID'  , bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay )
    net = tf_util.conv1d(l0_vertebra, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.conv1d(net, 64, 1, padding='VALID', bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    # net = tf.concat(axis=2, values=[net, meta_A])
    # net = tf_util.conv1d(net, 64, 1, padding='VALID', bn=True, is_training=is_training, scope='fc22', bn_decay=bn_decay)
    # net = tf_util.conv1d(net, 64, 1, padding='VALID', bn=True, is_training=is_training, scope='fc222', bn_decay=bn_decay)
    net = tf_util.conv1d(net, 3, 1, padding='VALID', activation_fn=None, scope='fc3')

    # Construct Unet model
    # net = tf_util.conv1d(l0_vertebra, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1',bn_decay=bn_decay)
    # net = tf_util.conv1d(net, 64, 1, padding='VALID', bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    # net = tf_util.conv1d(net, 24, 1, padding='VALID', activation_fn=None, scope='fc3')

    # if meta_A is not None:
    #     net = tf.concat(axis=2, values=[net, meta_A * FLAGS.metaWeight, meta_A * FLAGS.metaWeight, meta_A * FLAGS.metaWeight, meta_A * FLAGS.metaWeight])

    #net = unet.create_unet(l0_vertebra, train=False)
    # net = unet.create_unet(net, train=False)

    # displacements = tf.argmax(net, axis=3, name="y_pred")

    #net = tf.squeeze(net, [2])
    displacements = tf.sigmoid(net) * FLAGS.range_max * 2 - FLAGS.range_max
    return displacements

def get_displacements(input_points, input_vertebra_T, is_training, noise, meta_A, FLAGS, bn_decay=None):
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """

    # rigid_feautres = tf.concat(axis=1, values=[input_vertebra_T, input_vertebra_T, input_vertebra_T, input_vertebra_T,input_vertebra_T, input_vertebra_T, input_vertebra_T])
    #features = challismethod(rigid_feautres, rigid_feautres)
    #input_points = tf.concat(axis=1, values=[input_points, input_vertebra_T])

    batch_size = FLAGS.batch_size
    num_points = FLAGS.point_num

    point_cloud = input_points

    l0_xyz = point_cloud
    l0_points = None
    #l0_points = noise

    # Set Abstraction layers
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=200, radius=0.1 * FLAGS.radiusScal,
                                                       nsample=30, #64
                                                       mlp=[64, 64, 128], mlp2=None, group_all=False,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer1')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=100, radius=0.2 * FLAGS.radiusScal,
                                                       nsample=30,
                                                       mlp=[128, 128, 256], mlp2=None, group_all=False,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=50, radius=0.4 * FLAGS.radiusScal,
                                                       nsample=30,
                                                       mlp=[256, 256, 512], mlp2=None, group_all=False,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer3')

    # PointNet
    l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=None, radius=None, nsample=None,
                                                       mlp=[512, 512, 1024], mlp2=None, group_all=True,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer4')


    # l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=1024, radius=0.1 * FLAGS.radiusScal, nsample=64,
    #                                                    mlp=[64, 64, 128], mlp2=None, group_all=False,
    #                                                    is_training=is_training, bn_decay=bn_decay, scope='layer1')
    # l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=384, radius=0.2* FLAGS.radiusScal, nsample=64,
    #                                                    mlp=[128, 128, 256], mlp2=None, group_all=False,
    #                                                    is_training=is_training, bn_decay=bn_decay, scope='layer2')
    # l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=128, radius=0.4* FLAGS.radiusScal, nsample=64,
    #                                                    mlp=[256, 256, 512], mlp2=None, group_all=False,
    #                                                    is_training=is_training, bn_decay=bn_decay, scope='layer3')

    #PointNet
    # l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=None, radius=None, nsample=None,
    #                                                    mlp=[512, 512, 1024], mlp2=None, group_all=True,
    #                                                    is_training=is_training, bn_decay=bn_decay, scope='layer4')

    # Feature Propagation layers
    ### l4_points = pointnet_fp_module(l4_xyz, l5_xyz, l4_points, l5_points, [512,512], is_training, bn_decay, scope='fa_layer0')
    l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [512, 512], is_training, bn_decay, scope='fa_layer1')
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [512, 256], is_training, bn_decay, scope='fa_layer2')
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256, 128], is_training, bn_decay, scope='fa_layer3')
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128, 128, 128], is_training, bn_decay, scope='fa_layer4')

    # l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [512, 512], is_training, bn_decay, scope='fa_layer1')
    # l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [512, 256], is_training, bn_decay, scope='fa_layer2')
    # l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256, 128], is_training, bn_decay, scope='fa_layer3')
    # l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128, 128, 128], is_training, bn_decay, scope='fa_layer4')

    if noise is not None:
        l0_points = tf.concat(axis=2, values=[l0_points, noise])
        # meta_data = np.ndarray((1, 4), np.float32)
        # meta_data = tf.zeros(shape=[FLAGS.batch_size, FLAGS.point_num, FLAGS.metaLength], dtype=tf.float32, name=None)
    #
    # if meta_A is not None:
    # #     #meta_A = tf.concat(axis=1, values=[meta_A, input_vertebra_T])
    #     l0_points = tf.concat(axis=2, values=[l0_points, meta_A * FLAGS.metaWeight, meta_A * FLAGS.metaWeight, meta_A * FLAGS.metaWeight, meta_A * FLAGS.metaWeight])
    #    #l0_points = tf.concat(axis=2, values=[l0_points, meta_A * FLAGS.metaWeight])
        ## l0_points = tf.concat(axis=2, values=[l0_points, meta_A[0:1, 0:17, :]])
        ##l0_points = np.concatenate((l0_points, meta_A), axis=2)



    # net = tf_util.conv1d(l0_points, 128, 1, padding='VALID'  , bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay )
    # net = tf_util.conv1d(net, 64, 1, padding='VALID', bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    # net = tf_util.conv1d(net, 3, 1, padding='VALID', activation_fn=None, scope='fc3')
    # #net = tf_util.conv1d(net, 3, 6,  stride=6, padding='VALID', activation_fn=None, scope='fc3')


    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID'  , bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay )
    net = tf_util.conv1d(net, 64, 1, padding='VALID', bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util.conv1d(net, 3, 1, padding='VALID', activation_fn=None, scope='fc3')
    #net = tf_util.conv1d(net, 3, 6,  stride=6, padding='VALID', activation_fn=None, scope='fc3') # for good GPU to have kernel 6 for batch 6

    #displacements = tf.sigmoid(net) * FLAGS.range_max * 2 - FLAGS.range_max
    #displacements = tf.sigmoid(net)
    displacements = tf.sigmoid(net) * FLAGS.range_max * 2 - FLAGS.range_max

    return displacements

def get_displacements_AE(input_points, is_training, noise, meta_A, FLAGS, bn_decay=None):
# def get_displacements_AE(input_points, points_spinePre, points_spinePost, is_training, noise, meta_A, FLAGS,  bn_decay=None):
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """

    batch_size = FLAGS.batch_size
    num_points_trunk = FLAGS.point_num
    num_points_spine=108
    point_dim=3
    end_points = {}


    # if noise is not None:
    #   input_points = tf.concat(axis=2, values=[input_points, noise])


    # if meta_A is not None:
    #    # l0_vertebra = tf.concat(axis=2, values=[l0_vertebra, meta_A * FLAGS.metaWeight, meta_A * FLAGS.metaWeight, meta_A * FLAGS.metaWeight, meta_A * FLAGS.metaWeight])
    #   input_points = tf.concat(axis=2, values=[input_points, meta_A])

    input_image_trunk = tf.expand_dims(input_points, -1)
    # input_image_spine = tf.expand_dims(points_spinePost, -1)
    input_meta_A = tf.expand_dims(meta_A, -1)

    # Encoder Trunk
    net_trunk = tf_util.conv2d(input_image_trunk, 64, [1, point_dim],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    # net_trunk = tf_util.conv2d(net_trunk, 64, [1, 1],
    #                      padding='VALID', stride=[1, 1],
    #                      bn=True, is_training=is_training,
    #                      scope='conv2', bn_decay=bn_decay)
    # point_feat = tf_util.conv2d(net_trunk, 64, [1, 1],
    #                             padding='VALID', stride=[1, 1],
    #                             bn=True, is_training=is_training,
    #                             scope='conv3', bn_decay=bn_decay)
    net_trunk = tf_util.conv2d(net_trunk, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    # net_trunk = tf_util.conv2d(net_trunk, 512, [1, 1],
    #                      padding='VALID', stride=[1, 1],
    #                      bn=True, is_training=is_training,
    #                      scope='conv5', bn_decay=bn_decay)
    # global_feat_trunk = tf_util.max_pool2d(net_trunk, [num_points_trunk, 1], padding='VALID', scope='maxpool')

    # # Encoder Spine
    # net_spine = tf_util.conv2d(input_image_spine, 64, [1, point_dim],
    #                      padding='VALID', stride=[1, 1],
    #                      bn=True, is_training=is_training,
    #                      scope='conv11', bn_decay=bn_decay)
    # net_spine = tf_util.conv2d(net_spine, 64, [1, 1],
    #                      padding='VALID', stride=[1, 1],
    #                      bn=True, is_training=is_training,
    #                      scope='conv22', bn_decay=bn_decay)
    # point_feat = tf_util.conv2d(net_spine, 64, [1, 1],
    #                             padding='VALID', stride=[1, 1],
    #                             bn=True, is_training=is_training,
    #                             scope='conv33', bn_decay=bn_decay)
    # net_spine = tf_util.conv2d(point_feat, 128, [1, 1],
    #                      padding='VALID', stride=[1, 1],
    #                      bn=True, is_training=is_training,
    #                      scope='conv44', bn_decay=bn_decay)
    # # net_spine = tf_util.conv2d(net_spine, 512, [1, 1],
    # #                      padding='VALID', stride=[1, 1],
    # #                      bn=True, is_training=is_training,
    # #                      scope='conv55', bn_decay=bn_decay)
    # global_feat_spine = tf_util.max_pool2d(net_spine, [num_points_spine, 1],
    #                                  padding='VALID', scope='maxpool')

    # Encoder Meta
    # net_meta = tf_util.conv2d(input_meta_A, 64, [1, point_dim],
    #                            padding='VALID', stride=[1, 1],
    #                            bn=True, is_training=is_training,
    #                            scope='conv111', bn_decay=bn_decay)
    # net_meta = tf_util.conv2d(net_meta, 64, [1, 1],
    #                            padding='VALID', stride=[1, 1],
    #                            bn=True, is_training=is_training,
    #                            scope='conv222', bn_decay=bn_decay)
    # point_feat = tf_util.conv2d(net_meta, 64, [1, 1],
    #                             padding='VALID', stride=[1, 1],
    #                             bn=True, is_training=is_training,
    #                             scope='conv333', bn_decay=bn_decay)
    # net_meta = tf_util.conv2d(point_feat, 128, [1, 1],
    #                            padding='VALID', stride=[1, 1],
    #                            bn=True, is_training=is_training,
    #                            scope='conv444', bn_decay=bn_decay)
    # # net_spine = tf_util.conv2d(net_spine, 512, [1, 1],
    # #                      padding='VALID', stride=[1, 1],
    # #                      bn=True, is_training=is_training,
    # #                      scope='conv55', bn_decay=bn_decay)
    # global_feat_meta = tf_util.max_pool2d(net_meta, [num_points_spine, 1],
    #                                        padding='VALID', scope='maxpool')

    net_trunk = tf.reshape(net_trunk, [batch_size, -1])
    # net_trunk = tf.reshape(global_feat_trunk, [batch_size, -1])
    # net_spine = tf.reshape(global_feat_spine, [batch_size, -1])
    # net_meta = tf.reshape(global_feat_meta, [batch_size, -1])

    # net = tf.concat([net_trunk, net_spine], axis=1)
    # meta_A = tf.reshape(meta_A, [batch_size, -1])
    # net_trunk = tf.concat([net_trunk, meta_A], axis=1)
    # end_points['embedding'] = net
    # net = net_trunk

    # # FC Decoder
    # net = tf_util.fully_connected(net_trunk, 512, bn=True, is_training=is_training, scope='fc11', bn_decay=bn_decay)
    # net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc22', bn_decay=bn_decay)
    net = tf_util.fully_connected(net_trunk, 128, bn=True, is_training=is_training, scope='fc33', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, num_points_trunk * 3, activation_fn=None, scope='fc44')
    net = tf.reshape(net, (batch_size, num_points_trunk, 3))

    # #
    # # # Construct Conv model
    # net = net_trunk[:,:,-1,:]
    # # net = tf_util.conv1d(net, 256, 1, padding='VALID', bn=True, is_training=is_training, scope='fc0', bn_decay=bn_decay)
    # net = tf_util.conv1d(net, 128, 1, padding='VALID'  , bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay )
    # net = tf_util.conv1d(net, 64, 1, padding='VALID', bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    # net = tf_util.conv1d(net, 3, 1, padding='VALID', activation_fn=None, scope='fc3')

    displacements = tf.sigmoid(net) * FLAGS.range_max * 2 - FLAGS.range_max

    ###
    # if noise is not None:
    #     l0_points = tf.concat(axis=2, values=[l0_points, noise])
    #     # meta_data = np.ndarray((1, 4), np.float32)
    #     # meta_data = tf.zeros(shape=[FLAGS.batch_size, FLAGS.point_num, FLAGS.metaLength], dtype=tf.float32, name=None)
    # #
    # # if meta_A is not None:
    # #     l0_points = tf.concat(axis=2, values=[meta_A, l0_points])
    #     #l0_points = tf.concat(axis=2, values=[l0_points, meta_A * FLAGS.metaWeight, meta_A * FLAGS.metaWeight, meta_A * FLAGS.metaWeight, meta_A * FLAGS.metaWeight])
    #    #l0_points = tf.concat(axis=2, values=[l0_points, meta_A * FLAGS.metaWeight])
    #     ## l0_points = tf.concat(axis=2, values=[l0_points, meta_A[0:1, 0:17, :]])
    #     ##l0_points = np.concatenate((l0_points, meta_A), axis=2)
    #
    # # net = tf_util.conv1d(l0_points, 128, 1, padding='VALID'  , bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay )
    # # net = tf_util.conv1d(net, 64, 1, padding='VALID', bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    # # net = tf_util.conv1d(net, 3, 1, padding='VALID', activation_fn=None, scope='fc3')
    # # #net = tf_util.conv1d(net, 3, 6,  stride=6, padding='VALID', activation_fn=None, scope='fc3')
    #
    #
    # net = tf_util.conv1d(net, 128, 1, padding='VALID'  , bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay )
    # net = tf_util.conv1d(net, 64, 1, padding='VALID', bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    # net = tf_util.conv1d(net, 3, 1, padding='VALID', activation_fn=None, scope='fc3')
    # #net = tf_util.conv1d(net, 3, 6,  stride=6, padding='VALID', activation_fn=None, scope='fc3') # for good GPU to have kernel 6 for batch 6
    #
    # #displacements = tf.sigmoid(net) * FLAGS.range_max * 2 - FLAGS.range_max
    # #displacements = tf.sigmoid(net)
    # # displacements_1 = tf.sigmoid(net[:,0:30000,:]) * 300 * 2 - 300
    # # displacements_2 = tf.sigmoid(net[:,30000:60000,:]) * 50 * 2 - 50
    # # displacements= tf.concat(axis=1, values=[displacements_1, displacements_2])
    # displacements = tf.sigmoid(net) * FLAGS.range_max * 2 - FLAGS.range_max

    return displacements

def get_Accuracy_Rec(x, y, FLAGS, scope=None):
    with tf.op_scope([x, y], scope, 'pairwise_l2_norm2_batch'):
        # calculate shape loss
        diff = x-y
        square_diff = tf.square(diff)
        minRow = tf.reduce_sum(square_diff, axis=2)
        dis = tf.sqrt(minRow)
        minColl = tf.reduce_mean(dis, axis=1)
        Accuracy =tf.reduce_mean(minColl)

        return Accuracy

def get_Geometric_Loss_PointWise(predictedPts, targetpoints, FLAGS):
    # calculate shape loss
    diff = predictedPts - targetpoints
    square_diff = tf.square(diff)
    minRow = tf.reduce_sum(square_diff, axis=2)
    dis = tf.sqrt(minRow)
    # dis = tf.sqrt(1e-6 + minRow)
    minColl = tf.reduce_mean(dis, axis=1)
    shapeLoss = tf.reduce_mean(minColl)

    # square_dist = pairwise_l2_norm2_batch(targetpoints, predictedPts)
    # dist = tf.sqrt(square_dist)
    #
    # # calculate density loss
    # square_dist2 = pairwise_l2_norm2_batch(targetpoints, targetpoints)
    # dist2 = tf.sqrt(square_dist2)
    # knndis = tf.nn.top_k(tf.negative(dist), k=FLAGS.nnk)
    # knndis2 = tf.nn.top_k(tf.negative(dist2), k=FLAGS.nnk)
    # densityLoss = tf.reduce_mean(tf.abs(knndis.values - knndis2.values))

    # calculate rigid transformation loss
    # square_dist3 = pairwise_l2_norm2_batch(targetpoints, predictedPts)
    # dist3 = tf.sqrt(square_dist3)
    # minRow3 = tf.reduce_min(dist3, axis=2)
    # minCol3 = tf.reduce_min(dist3, axis=1)
    # shapeLoss = tf.reduce_mean(minRow) + tf.reduce_mean(minCol)

    #data_loss = shapeLoss + densityLoss * FLAGS.densityWeight
    densityLoss=tf.constant(0.0, dtype=tf.float32)
    return  shapeLoss, densityLoss


def get_Geometric_Loss(predictedPts, targetpoints, FLAGS):

    # calculate shape loss
    square_dist = pairwise_l2_norm2_batch(targetpoints, predictedPts)
    dist = tf.sqrt( square_dist )
    minRow = tf.reduce_min(dist, axis=2)
    minCol = tf.reduce_min(dist, axis=1)
    shapeLoss = tf.reduce_mean(minRow) + tf.reduce_mean(minCol)
    # shapeLoss = tf.reduce_mean(1e-5 +minRow) + tf.reduce_mean(1e-5 +minCol)
	

    # calculate density loss
    # square_dist2 = pairwise_l2_norm2_batch(targetpoints, targetpoints)
    # dist2 = tf.sqrt(square_dist2)
    # knndis = tf.nn.top_k(tf.negative(dist), k=FLAGS.nnk)
    # knndis2 = tf.nn.top_k(tf.negative(dist2), k=FLAGS.nnk)
    # densityLoss = tf.reduce_mean(tf.abs(knndis.values - knndis2.values))

    # calculate rigid transformation loss
    #square_dist3 = pairwise_l2_norm2_batch(targetpoints, predictedPts)
    #dist3 = tf.sqrt(square_dist3)
    #minRow3 = tf.reduce_min(dist3, axis=2)
    #minCol3 = tf.reduce_min(dist3, axis=1)
    #shapeLoss = tf.reduce_mean(minRow) + tf.reduce_mean(minCol)

    densityLoss = tf.constant(0.0, dtype=tf.float32)
    # data_loss = shapeLoss + densityLoss * FLAGS.densityWeight
    # return data_loss, shapeLoss, densityLoss
    return  shapeLoss, densityLoss



def get_Regularizing_Loss(pointSet_A_ph, pointSet_B_ph,  Predicted_A, Predicted_B):

    displacements_A = tf.concat(axis=2, values=[pointSet_A_ph, Predicted_B])
    displacements_B = tf.concat(axis=2, values=[Predicted_A, pointSet_B_ph])

    square_dist = pairwise_l2_norm2_batch(displacements_A, displacements_B)
    dist = tf.sqrt(square_dist)

    minRow = tf.reduce_min(dist, axis=2)
    minCol = tf.reduce_min(dist, axis=1)
    RegularLoss = (tf.reduce_mean(minRow) + tf.reduce_mean(minCol)) / 2

    return RegularLoss

# def get_Local_Loss(pointSet_A_ph, targetpoints, Predicted_A):
#
#     displacements_real = pairwise_l2_norm2_batch(pointSet_A_ph, targetpoints)
#     dist_real = tf.sqrt(displacements_real)
#     displacements_predicted = pairwise_l2_norm2_batch(pointSet_A_ph, Predicted_A)
#     dist_predicted = tf.sqrt(displacements_predicted)
#
#     square_dist = pairwise_l2_norm2_batch( dist_real,   dist_predicted )
#     dist = tf.sqrt(square_dist)
#
#     minRow = tf.reduce_min(dist, axis=2)
#     minCol = tf.reduce_min(dist, axis=1)
#     localLoss = (tf.reduce_mean(minRow) + tf.reduce_mean(minCol))/2
#
#     return localLoss


def pairwise_l2_norm2_batch(x, y, scope=None):
    with tf.op_scope([x, y], scope, 'pairwise_l2_norm2_batch'):
        nump_x = tf.shape(x)[1]
        nump_y = tf.shape(y)[1]

        xx = tf.expand_dims(x, -1)
        xx = tf.tile(xx, tf.stack([1, 1, 1, nump_y]))

        yy = tf.expand_dims(y, -1)
        yy = tf.tile(yy, tf.stack([1, 1, 1, nump_x]))
        yy = tf.transpose(yy, perm=[0, 3, 2, 1])

        diff = tf.subtract(xx, yy)
        square_diff = tf.square(1e-4 + diff)

        square_dist = tf.reduce_sum(square_diff, 2)

        return square_dist

def pairwise_l2_norm2_batch_high_dim(x, y, scope=None):
    with tf.op_scope([x, y], scope, 'pairwise_l2_norm2_batch'):
        nump_x = tf.shape(x)[1]
        nump_y = tf.shape(y)[1]

        xx = tf.expand_dims(x, -1) #xxx = tf.expand_dims(xx, -1)
        xx = tf.tile(xx, tf.stack([1, 1, 1, 1, nump_y])) #      xx = tf.tile(xxx, tf.stack([1, 1, 1, 1, nump_y, nump_y]))

        yy = tf.expand_dims(y, -1)
        yy = tf.tile(yy, tf.stack([1, 1, 1, 1, nump_x]))
        yy = tf.transpose(yy, perm=[0, 4, 2, 3, 1])

        diff = tf.subtract(xx, yy)
        square_diff = tf.square(diff)

        square_dist = tf.reduce_sum(square_diff, [2 , 3])

        return square_dist

def f(x):

    return x

def rigid_transform_3D(Predicted_shape, target_T, target_R, FLAGS, scope=None):
    #with tf.op_scope([Predicted_shape], scope, 'rigid_transform_3D'): # A: shape(102,3)
         rigidLoss = tf.constant(2.0, dtype=tf.float32)
     #if FLAGS.modeCode == "create":
       ###sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
       ###A_array_all = np.array(Predicted_post.eval())
       ###from tensorflow.python.keras import backend as K
       ###sess = K.get_session()
       ###A_array_all = sess.run(Predicted_post)
       ###sess.close()
       #init_g = tf.global_variables_initializer()
       #init_l = tf.local_variables_initializer()
       #Predicted_shape_ = np.zeros((2, 3, 102, 3), dtype=np.float32)
       #A_array_all = np.zeros((3, 102, 3), dtype=np.float32)

       ####A_array_all = tf.make_ndarray(Predicted_shape)

       ### A_array_all = tf.keras.backend.eval(Predicted_shape)
       ### sess = backend.get_session()
       ### A_array_all = sess.run(Predicted_shape)
       ### target_T = sess.run(target_T)
       ### target_R = sess.run(target_R)
       #with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess2:
           #sess2.run(init_g)
           #sess2.run(init_l)
           #Predicted_shape_ = Predicted_shape.eval(session=sess2)
           #Predicted_shape_ = sess2.run(Predicted_shape)
           #target_T = sess2.run(target_T)
           #target_R = sess2.run(target_R)
           #A_array_all = np.array(Predicted_shape_)
           ### saver = tf.train.Saver(max_to_keep=5)
           ### saver.save(sess2, os.path.join('utput_airplane_skeleton - surface / trained_models'))
           ### saver.restore(sess2, tf.train.latest_checkpoint(os.path.join('utput_airplane_skeleton - surface / trained_models')))
           ###A_array_all= tf.keras.backend.eval(Predicted_shape)
           ### sess = backend.get_session()
           ### array = sess.run(Predicted_shape_)


       ### sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
       ### sess.run(init_g)
       ### sess.run(init_l)
       ### Predicted_shape_ = sess.run(Predicted_shape)
       ### target_T = sess.run(target_T)
       ### target_R = sess.run(target_R)
       ### A_array_all = np.array(Predicted_shape_)
       ### print('\tData_loss_A1 = %.4f,' % Predicted_shape_[0, 0, 0])
       ### print('\tData_loss_A2 = %.4f,' % Predicted_shape_[0, 0, 1])
       ### print('\tData_loss_A3 = %.4f,' % Predicted_shape_[0, 0, 2])
       ### rigidLoss = tf.constant(3.0, dtype=tf.float32)
       ### init_g = tf.global_variables_initializer()
       ### init_l = tf.local_variables_initializer()
       ### with tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
       ###      sess.run(init_g)
       ###      sess.run(init_l)
       ###      Predicted_shape = sess.run(Predicted_shape)
       ### proto_tensor = tf.make_tensor_proto(Predicted_shape[0,:,:])  # convert `tensor a` to a proto tensor
       ### A_array_all=tf.make_ndarray(Predicted_shape)
       ### A_array_all = np.array(Predicted_shape)

       #if A_array_all.ndim != 0:
         ###rigidLoss = tf.constant(4.0, dtype=tf.float32)
         list_translations =[]
         list_rotations = []
         batch_size = FLAGS.batch_size #batch_size = len(A_array_all)  # FLAGS.batch_size
         translations = tf.zeros((batch_size, 23, 3), dtype=np.float32) #translations = np.zeros((batch_size, 16, 3), dtype=np.float32)
         rotations = tf.zeros((batch_size, 23, 3), dtype=np.float32)
         #rotations = tf.zeros((batch_size, 16, 3, 3), dtype=np.float32) #rotations = np.zeros((batch_size, 16, 3), dtype=np.float32)
         for batch_id in range(0, batch_size, 1):
           slice1_ = tf.slice(Predicted_shape, [batch_id, 0, 0], [1, -1, -1]) #A_array = np.array(A_array_all[batch_id, :,:])
           ###A_array = np.array(Predicted_shape[0, :, :])
           #### T_array = np.array(target_T)
           #### R_array = np.array(target_R)
           c = tf.zeros((23, 3), dtype=np.float32) #c = np.zeros((17, 3))
           vecZ = tf.zeros((3), dtype=np.float32) #vecZ = np.zeros(3)
           vecY = tf.zeros((3), dtype=np.float32) #vecY = np.zeros(3)
           vecX = tf.zeros((3), dtype=np.float32) #vecX = np.zeros(3)
           vec = tf.zeros((batch_size, 24, 3), dtype=np.float32) #vec = np.zeros((17, 3, 3))
           trans = tf.zeros((23, 3), dtype=np.float32) #trans = np.zeros((16, 3), dtype=np.float32)
           rotate = tf.zeros((23, 3), dtype=np.float32) #rotate = np.zeros((16, 3), dtype=np.float32)
           row=102 #row = len(A_array) #slice1_.shape[1].value
           if row == 102:
             # ref_x=(slice1_[0,101,0] + slice1_[0,100,0]) /2 # x = (A_array[101, 0] + A_array[ 100, 0]) / 2
             # ref_y = (slice1_[0, 101, 1] + slice1_[0, 100, 1]) / 2 # y = (A_array[101, 1] + A_array[100, 1]) / 2
             # ref_z=(slice1_[0,101,2] + slice1_[0,100,2]) /2 # z = (A_array[101, 2] + A_array[100, 2]) / 2
             # ref_t=tf.stack([ref_x,ref_y,ref_z],0) # ref = np.array([x, y, z])
             j = 0
             list_c = []
             list_xyz = []
             list_vec = []
             # Translation between all points
             for i in range(0, 97, 6):
                 T_1 = (slice1_[0, i + 6, :] - slice1_[0, i,
                                               :])  # / 2 - ref_t[0]  # c[j,0]=(A_array[i+5,0]+A_array[i+4,0])/2-ref[0]
                 T_2 = (slice1_[0, i + 7, :] - slice1_[0, i + 1,
                                               :])  # / 2 - ref_t[1]  # c[j,1]=(A_array[i+5,1]+A_array[i+4,1])/2-ref[1]
                 T_3 = (slice1_[0, i + 8, :] - slice1_[0, i + 2,
                                               :])  # / 2 - ref_t[2]  # c[j,2]=(A_array[i+5,2]+A_array[i+4,2])/2-ref[2]
                 T_4 = (slice1_[0, i + 9, :] - slice1_[0, i + 3, :])  # / 2 - ref_t[2]
                 T_5 = (slice1_[0, i + 10, :] - slice1_[0, i + 4, :])  # / 2 - ref_t[2]
                 T_6 = (slice1_[0, i + 11, :] - slice1_[0, i + 5, :])  # / 2 - ref_t[2]
                 list_c.append(T_1)
                 list_c.append(T_2)
                 list_c.append(T_3)
                 list_c.append(T_4)
                 list_c.append(T_5)
                 list_c.append(T_6)

             for i in range(0, 108, 6):
                list_xyz = []
                ### center
                # c_x=(slice1_[0,i+5,0] + slice1_[0,i+4,0]) /2 -ref_t[0] # c[j,0]=(A_array[i+5,0]+A_array[i+4,0])/2-ref[0]
                # c_y=(slice1_[0,i+5,1] + slice1_[0,i+4,1]) /2 -ref_t[1] # c[j,1]=(A_array[i+5,1]+A_array[i+4,1])/2-ref[1]
                # c_z=(slice1_[0,i+5,2] + slice1_[0,i+4,2]) /2 -ref_t[2] # c[j,2]=(A_array[i+5,2]+A_array[i+4,2])/2-ref[2]
                # c_t=tf.concat([[c_x], [c_y], [c_z]],0)
                # list_c.append(c_t)
                # ###first local axes
                # list_xyz = []
                # ## axe_Z
                vecZ_x=slice1_[0,i+5,0] - slice1_[0,i+4,0] # vec[j,0,0]=vecZ[0]=A_array[i+5,0]-A_array[i+4,0]
                vecZ_y=slice1_[0,i+5,1] - slice1_[0,i+4,1]  # vec[j,0,1]=vecZ[1]=A_array[i+5,1]-A_array[i+4,1]
                vecZ_z=slice1_[0,i+5,2] - slice1_[0,i+4,2]  # vec[j,0,2]=vecZ[2]=A_array[i+5,2]-A_array[i+4,2]
                vecZ_t=tf.concat([[vecZ_x], [vecZ_y], [vecZ_z]],0)
                # ###vecZ=vecZ./norm(vecZ)
                if tf.norm(vecZ_t, axis=None) == 0:
                   vecZ_norm_t = vecZ_t
                else:
                   vecZ_norm_t = vecZ_t / tf.norm(vecZ_t, axis=None)  # vecZ_norm = vecZ/np.linalg.norm(vecZ)
                # ### axe_Y
                vecY_x=(slice1_[0,i+1,0] - slice1_[0,i+0,0] +  slice1_[0,i+3,0]-slice1_[0,i+2,0])/2 # vec[j,1,0]=vecY[0]=(A_array[i+1,0]-A_array[i+0,0] + A_array[i+3,0]-A_array[i+2,0])/2
                vecY_y=(slice1_[0,i+1,1] - slice1_[0,i+0,1] +  slice1_[0,i+3,1]-slice1_[0,i+2,1])/2 # vec[j,1,1]=vecY[1]=(A_array[i+1,1]-A_array[i+0,1] + A_array[i+3,1]-A_array[i+2,1])/2
                vecY_z=(slice1_[0,i+1,2] - slice1_[0,i+0,2] +  slice1_[0,i+3,2]-slice1_[0,i+2,2])/2 # vec[j,1,2]=vecY[2]=(A_array[i+1,2]-A_array[i+0,2] + A_array[i+3,2]-A_array[i+2,2])/2
                vecY_t=tf.concat([[vecY_x], [vecY_y], [vecY_z]],0)
                # ###[vecY]= gramschimdit(vecZ,vecY)
                vecY_norm_t =  gramschimdit3(vecZ_norm_t, vecY_t) # vecY_norm = gramschimdit3(vecZ_norm, vecY)
                # ### axe_X
                # ###vecX_x=1 # vec[j,2,0]=vecX[0]=1
                # ###vecX_y=1 # vec[j,2,1]=vecX[1]=1
                # ###vecX_z=1 # vec[j,2,2]=vecX[2]=1
                # ###vecX_t = tf.stack([vecX_x, vecX_y, vecX_z], 0)
                vecX_t=tf.constant([1,1,1],dtype=tf.float32)
                vecZY_t = tf.stack((vecZ_norm_t, vecY_norm_t)) #vecZY = np.stack((vecZ_norm, vecY_norm))
                # ###[vecX]= gramschimdit2([vecZ vecY], vecX)
                vecX_norm_t = gramschimdit3(vecZY_t, vecX_t)
                # ###vec[j, 0 ,:] = vecZ_norm
                # ###vec[j, 1, :] = vecY_norm
                # ###vec[j, 2, :] = vecX_norm
                j = j + 1
                list_xyz.append(vecZ_norm_t)
                list_xyz.append(vecY_norm_t)
                list_xyz.append(vecX_norm_t)
                list_vec.append(list_xyz)
                # vec_R_1=tf.stack(list_xyz)
                #
                # ####second local axes
                # list_xyz = []
                # ## axe_Z
                # vecZ_x = slice1_[0, i + 10, 0] - slice1_[0, i + 11, 0]  # vec[j,0,0]=vecZ[0]=A_array[i+5,0]-A_array[i+4,0]
                # vecZ_y = slice1_[0, i + 10, 1] - slice1_[0, i + 11, 1]  # vec[j,0,1]=vecZ[1]=A_array[i+5,1]-A_array[i+4,1]
                # vecZ_z = slice1_[0, i + 10, 2] - slice1_[0, i + 11, 2]  # vec[j,0,2]=vecZ[2]=A_array[i+5,2]-A_array[i+4,2]
                # vecZ_t = tf.concat([[vecZ_x], [vecZ_y], [vecZ_z]], 0)
                # ###vecZ=vecZ./norm(vecZ)
                # if tf.norm(vecZ_t, axis=None)==0:
                #  vecZ_norm_t=vecZ_t
                # else:
                #   vecZ_norm_t = vecZ_t / tf.norm(vecZ_t, axis=None)  # vecZ_norm = vecZ/np.linalg.norm(vecZ)
                # ### axe_Y
                # vecY_x = (slice1_[0, i + 7, 0] - slice1_[0, i + 6, 0] + slice1_[0, i + 9, 0] - slice1_[
                #     0, i + 8, 0]) / 2  # vec[j,1,0]=vecY[0]=(A_array[i+1,0]-A_array[i+0,0] + A_array[i+3,0]-A_array[i+2,0])/2
                # vecY_y = (slice1_[0, i + 7, 1] - slice1_[0, i + 6, 1] + slice1_[0, i + 9, 1] - slice1_[
                #     0, i + 8, 1]) / 2  # vec[j,1,1]=vecY[1]=(A_array[i+1,1]-A_array[i+0,1] + A_array[i+3,1]-A_array[i+2,1])/2
                # vecY_z = (slice1_[0, i + 7, 2] - slice1_[0, i + 6, 2] + slice1_[0, i + 9, 2] - slice1_[
                #     0, i + 8, 2]) / 2  # vec[j,1,2]=vecY[2]=(A_array[i+1,2]-A_array[i+0,2] + A_array[i+3,2]-A_array[i+2,2])/2
                # vecY_t = tf.concat([[vecY_x], [vecY_y], [vecY_z]], 0)
                # ###[vecY]= gramschimdit(vecZ,vecY)
                # vecY_norm_t=gramschimdit3(vecZ_norm_t, vecY_t)  # vecY_norm = gramschimdit3(vecZ_norm, vecY)
                # ### axe_X
                # ###vecX_x=1 # vec[j,2,0]=vecX[0]=1
                # ###vecX_y=1 # vec[j,2,1]=vecX[1]=1
                # ###vecX_z=1 # vec[j,2,2]=vecX[2]=1
                # ###vecX_t = tf.stack([vecX_x, vecX_y, vecX_z], 0)
                # vecX_t = tf.constant([1, 1, 1], dtype=tf.float32)
                # vecZY_t = tf.stack((vecZ_norm_t, vecY_norm_t))  # vecZY = np.stack((vecZ_norm, vecY_norm))
                # ###[vecX]= gramschimdit2([vecZ vecY], vecX)
                # vecX_norm_t=gramschimdit3(vecZY_t, vecX_t)  # vecX_norm = gramschimdit3(vecZY, vecX)
                #
                # ###vec[j, 0 ,:] = vecZ_norm
                # ###vec[j, 1, :] = vecY_norm
                # ###vec[j, 2, :] = vecX_norm
                # #j = j + 1
                # list_xyz.append(vecZ_norm_t)
                # list_xyz.append(vecY_norm_t)
                # list_xyz.append(vecX_norm_t)
                # # list_vec.append(list_xyz)
                # vec_R_2 = tf.stack(list_xyz)
                # ###rottaion between two local axes
                # R = challismethod(vec_R_1,vec_R_2)
                # list_vec.append(R)
                # list_vec.append(R)
                # list_vec.append(R)
                # list_vec.append(R)
                # list_vec.append(R)
                # list_vec.append(R)


           centers_t=tf.stack(list_c)
           vec_t=tf.stack(list_vec)

           # for translation all centers only should be activated
           list_trans=[]
           list_rotate=[]
           for index in range(0, 17, 1):
              # list_trans.append(centers_t[index+1,:] - centers_t[index,:]) # trans[index]= c[index+1,:] - c[index,:]
              # # #rotate[index] = challismethod(np.reshape(vec[index,:,:],(3,3)),np.reshape(vec[index+1,:,:],(3,3)))
              R= challismethod(tf.reshape(vec_t[index,:,:],[3,3]),tf.reshape(vec_t[index+1,:,:],[3,3]))
              list_rotate.append(R)

           # list_rotate.append(mat2vec(vec_t[index+1,:,:]))
           trans = tf.stack(centers_t)
           # rotate = tf.stack(vec_t)
           #trans=tf.stack(list_trans)
           rotate=tf.stack(list_rotate)
           list_translations.append(trans) # translations[batch_id, :, :] = trans[:, :]
           list_rotations.append(rotate) # rotations[batch_id, :, :] = rotate[:, :]


         translations=tf.stack(list_translations)
         rotations=tf.stack(list_rotations)
         ####translations_t = tf.convert_to_tensor(translations,np.float32)
         ##rotations_t = tf.convert_to_tensor(rotations,np.float32)
         ###T_array = tf.convert_to_tensor(T_array,np.float32)
         ###R_array = tf.convert_to_tensor(R_array,np.float32)
         ##calculate translation loss

         ### p = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, 16, 3))
         ### with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess2:
         ###     sess2.run(init_g)
         ###     sess2.run(init_l)
         ###     #h = tf.get_session_handle(translations_t)
         ###     h = sess2.run(translations_t)  # gives you a handle
         ###     sess2.run(p, feed_dict={p: h})



         # square_dist = pairwise_l2_norm2_batch(target_T, translations)
         # dist = tf.sqrt(square_dist)
         # minRow = tf.reduce_min(dist, axis=2)
         # minCol = tf.reduce_min(dist, axis=1)
         # translationLoss = tf.reduce_mean(minRow) + tf.reduce_mean(minCol)
         #diff = target_T[:,6:22,:] - translations
         diff = target_T - translations
         square_diff = tf.square(diff)
         minRow = tf.reduce_sum(square_diff, axis=2)
         dis = tf.sqrt(minRow)
         minColl = tf.reduce_mean(dis, axis=1)
         translationLoss = tf.reduce_mean(minColl)


         ### square_dist2 = pairwise_l2_norm2_batch(target_R, rotations)
         ### dist2 = tf.sqrt(square_dist2)
         ### minRow = tf.reduce_min(dist2, axis=2)
         ### minCol = tf.reduce_min(dist2, axis=1)
         ### rotationLoss = tf.reduce_mean(minRow) + tf.reduce_mean(minCol)

         diff = target_R[:,6:23,:] - rotations
         square_diff = tf.square(diff)
         minRow = tf.reduce_sum(square_diff, axis=2)
         #dis = tf.sqrt(minRow)
         dis = tf.sqrt(1e-4 + minRow)
         minColl = tf.reduce_mean(dis, axis=1)
         rotationLoss = tf.reduce_mean(minColl)

# calculate rigid loss
         rigidLoss = rotationLoss + translationLoss
         # rigidLoss = translationLoss
         return rigidLoss


def challismethod(X, Y, scope=None):
    rowsX=X.shape[0]
    colsX=X.shape[1]
    rowsY=Y.shape[0]
    colsY=Y.shape[1]
    X_transpose=tf.transpose(X) # X_transpose=np.transpose(X)
    C=1/colsX.value*Y*X_transpose # C=1/colsX*Y*X_transpose
    with tf.device('/cpu:0'):
     U, S, V = tf.linalg.svd(C)
    #U, V = tf.linalg.eigh(C)
    #U, S, V = np.linalg.svd(C, full_matrices=True) #U, V = tf.linalg.eigh(C)  #U, S, V = tf.linalg.svd(C) #U, S, V = np.linalg.svd(C, full_matrices=True)
    #U, S, V = gradient_svd(C)
    V_transpose =tf.transpose(V) # V_transpose =np.transpose(V)
    det=tf.linalg.det(U*V_transpose) # det=np.linalg.det(U*V_transpose)
    ### a = np.array([[1, 0, 0]])
    ### b = np.array([[0, 1, 0]])
    ### c = np.array([[0, 0, det]])
    M=tf.stack((tf.constant([1,0,0],dtype=tf.float32), tf.constant([0,1,0],dtype=tf.float32), [0, 0, det]), axis=1) # M=np.stack(([1, 0, 0], [0, 1, 0], [0, 0, det]), axis=1)
    #R_mat = U * M * V_transpose (matlab code)??
    R_mat = U * V_transpose
    R_vec = mat2vec(R_mat)
    return R_vec
    #return R_mat
#
def gramschimdit3(axes,axeinit):
    # axe = np.zeros(3)
    # a = sum(np.multiply(axeinit, axes))
    # b = np.power((np.linalg.norm(axes)), 2)
    # axe = axe - a / b
    # axe = np.multiply(axe, axes)
   axe=axeinit
   if axes.shape[0]==1: # if axes.shape[0]==1:
       a = tf.reduce_sum(tf.multiply(axeinit, axes)) # a = sum(tf.multiply(axeinit, axes))
       b = tf.power((tf.norm(axes)), 2) #    b = tf.power((np.linalg.norm(axes)), 2)
       axe = axe - a / b
       axe = tf.multiply(axe, axes)
   else:
    #for i in range(axes.ndim):
     for i in range(axes.shape[0]):
       a=tf.reduce_sum(tf.multiply(axeinit,axes[i])) # a=sum(tf.multiply(axeinit,axes[i]))
       b= tf.pow((tf.norm(axes[i])),2) #b= tf.power((np.linalg.norm(axes[i])),2)
       axe=axe-a/b
       axe= tf.multiply(axe,axes[i])
    #axe = axe - (np.sum(axeinit. * axes(:, i)) / (norm(axes(:, i)) ^ 2)).*axes(:, i);
   return axe/tf.norm(axe) #axe/np.linalg.norm(axe)

def gramschimdit2(vectors):
    basis = []
    for v in vectors:
        w = v - np.sum( np.dot(v,b) * b for b in basis )
        if (w > 1e-10).any():
            basis.append(w/np.linalg.norm(w))
    return np.array(basis)

def gramschimdit(X):
    Y = []
    for i in range(len(X)):
        temp_vec = X[i]
        for inY in Y:
            proj_vec = proj(inY, X[i])
            # print "i =", i, ", projection vector =", proj_vec
            temp_vec = list(map(lambda x, y: x - y, temp_vec, proj_vec))
            # print "i =", i, ", temporary vector =", temp_vec
        Y.append(temp_vec)
    return Y

def gs_cofficient(v1, v2):
    return np.dot(v2, v1) / np.dot(v1, v1)

def multiply(cofficient, v):
    a = list(map((lambda x: x * cofficient), v))
    return a

def proj(v1, v2):
    return multiply(gs_cofficient(v1, v2) , v1)

def mat2vec(v1):
    input = (tf.linalg.trace(v1)-1)/2 # input = (np.trace(v1)-1)/2
    # if input < -1:
    #     input = -1
    # elif input > 1:
    #     input = 1
    angle = tf.acos(input)   # angle =np.arccos(input)
    if angle!=0 :
        matrix=(v1-tf.transpose(v1))/(2*tf.sin(angle)) #matrix=(v1-np.transpose(v1))/(2*np.sin(angle))
    else:
        matrix=tf.zeros((3,3),dtype=tf.float32) # matrix=np.zeros((3,3),dtype=np.float32)

    axe = tf.stack((matrix[2,1], matrix[0,2], matrix[1,0]), axis=0) #axe = np.stack((matrix[2,1], matrix[0,2], matrix[1,0]), axis=0)
    vec = axe * angle
    return vec


@ops.RegisterGradient("Svd")
def _SvdGrad(op, grad_s, grad_u, grad_v):
    """Gradient for the singular value decomposition."""

    # The derivation for the compute_uv=False case, and most of
    # the derivation for the full_matrices=True case, are in
    # Giles' paper (see reference at top of file).  A derivation for
    # the full_matrices=False case is available at
    # https://j-towns.github.io/papers/svd-derivative.pdf
    # The derivation for complex valued SVD can be found in
    # https://re-ra.xyz/misc/complexsvd.pdf or
    # https://giggleliu.github.io/2019/04/02/einsumbp.html
    a = op.inputs[0]
    a_shape = a.get_shape().with_rank_at_least(2)
    grad_s = math_ops.cast(grad_s, a.dtype)
    grad_s_mat = array_ops.matrix_diag(grad_s)

    if not op.get_attr("compute_uv"):
        s, u, v = linalg_ops.svd(a, compute_uv=True)
        grad_a = math_ops.matmul(u, math_ops.matmul(grad_s_mat, v, adjoint_b=True))
        grad_a.set_shape(a_shape)
        return grad_a

    full_matrices = op.get_attr("full_matrices")

    grad_u_shape = grad_u.get_shape().with_rank_at_least(2)
    grad_v_shape = grad_v.get_shape().with_rank_at_least(2)
    m = a_shape.dims[-2].merge_with(grad_u_shape[-2])
    n = a_shape.dims[-1].merge_with(grad_v_shape[-2])
    batch_shape = a_shape[:-2].merge_with(grad_u_shape[:-2]).merge_with(
        grad_v_shape[:-2])
    a_shape = batch_shape.concatenate([m, n])

    m = a_shape.dims[-2].value
    n = a_shape.dims[-1].value
    # TODO(rmlarsen): Make this work with placeholders.
    if m is None or n is None:
        raise NotImplementedError(
            "SVD gradient has not been implemented for input with unknown "
            "inner matrix shape.")

    s = op.outputs[0]
    u = op.outputs[1]
    v = op.outputs[2]
    s = math_ops.cast(s, a.dtype)

    use_adjoint = False
    if m > n:
        # Compute the gradient for A^H = V * S^T * U^H, and (implicitly) take the
        # Hermitian transpose of the gradient at the end.
        use_adjoint = True
        m, n = n, m
        u, v = v, u
        grad_u, grad_v = grad_v, grad_u

    with ops.control_dependencies([grad_s, grad_u, grad_v]):
        if full_matrices and abs(m - n) > 1:
            raise NotImplementedError(
                "svd gradient is not implemented for abs(m - n) > 1 "
                "when full_matrices is True")
        s_mat = array_ops.matrix_diag(s)
        s2 = math_ops.square(s)

        # NOTICE: Because of the term involving f, the gradient becomes
        # infinite (or NaN in practice) when singular values are not unique.
        # Mathematically this should not be surprising, since for (k-fold)
        # degenerate singular values, the corresponding singular vectors are
        # only defined up a (k-dimensional) subspace. In practice, this can
        # lead to numerical instability when singular values are close but not
        # exactly equal.
        # To avoid nan in cases with degenrate sigular values or zero sigular values
        # in calculating f and s_inv_mat, we introduce a Lorentz brodening.

        def safe_reciprocal(x, epsilon=1E-20):
            return x * math_ops.reciprocal(x * x + epsilon)

        s_shape = array_ops.shape(s)
        f = array_ops.matrix_set_diag(
            safe_reciprocal(
                array_ops.expand_dims(s2, -2) - array_ops.expand_dims(s2, -1)
            ), array_ops.zeros_like(s))
        s_inv_mat = array_ops.matrix_diag(safe_reciprocal(s))

        v1 = v[..., :, :m]
        grad_v1 = grad_v[..., :, :m]

        u_gu = math_ops.matmul(u, grad_u, adjoint_a=True)
        v_gv = math_ops.matmul(v1, grad_v1, adjoint_a=True)

        f_u = f * u_gu
        f_v = f * v_gv

        term1_nouv = (
                grad_s_mat + math_ops.matmul(f_u + _adjoint(f_u), s_mat) +
                math_ops.matmul(s_mat, f_v + _adjoint(f_v)))

        term1 = math_ops.matmul(u, math_ops.matmul(term1_nouv, v1, adjoint_b=True))

        if m == n:
            grad_a_before_transpose = term1
        else:
            gv1t = array_ops.matrix_transpose(grad_v1, conjugate=True)
            gv1t_v1 = math_ops.matmul(gv1t, v1)
            term2_nous = gv1t - math_ops.matmul(gv1t_v1, v1, adjoint_b=True)

            if full_matrices:
                v2 = v[..., :, m:n]
                grad_v2 = grad_v[..., :, m:n]

                v1t_gv2 = math_ops.matmul(v1, grad_v2, adjoint_a=True)
                term2_nous -= math_ops.matmul(v1t_gv2, v2, adjoint_b=True)

            u_s_inv = math_ops.matmul(u, s_inv_mat)
            term2 = math_ops.matmul(u_s_inv, term2_nous)

            grad_a_before_transpose = term1 + term2

        if a.dtype.is_complex:
            eye = _linalg.eye(s_shape[-1], batch_shape=s_shape[:-1], dtype=a.dtype)
            l = eye * v_gv
            term3_nouv = math_ops.matmul(s_inv_mat, _adjoint(l) - l)
            term3 = 1 / 2. * math_ops.matmul(u, math_ops.matmul(term3_nouv, v1, adjoint_b=True))

            grad_a_before_transpose += term3

        if use_adjoint:
            grad_a = array_ops.matrix_transpose(grad_a_before_transpose, conjugate=True)
        else:
            grad_a = grad_a_before_transpose

        grad_a.set_shape(a_shape)
        return grad_a


def _LeftShift(x):
    """Shifts next-to-last dimension to the left, adding zero on the right."""
    rank = array_ops.rank(x)
    zeros = array_ops.zeros((rank - 2, 2), dtype=dtypes.int32)
    pad = array_ops.concat([zeros, array_ops.constant([[0, 1], [0, 0]])], axis=0)
    return array_ops.pad(x[..., 1:, :], pad)


def _RightShift(x):
    """Shifts next-to-last dimension to the right, adding zero on the left."""
    rank = array_ops.rank(x)
    zeros = array_ops.zeros((rank - 2, 2), dtype=dtypes.int32)
    pad = array_ops.concat([zeros, array_ops.constant([[1, 0], [0, 0]])], axis=0)
    return array_ops.pad(x[..., :-1, :], pad)

def _adjoint(x):
    # Transposes the last two dimensions of and conjugates tensor matrix
    adj_X = tf.transpose(x) # tf.transpose(x, perm=[0,2,1]) is short for dim_0 -> dim_0, dim_1 -> dim_2, dim_2 -> dim_1
    # if tf.linalg.det(x) ==0:
    #     adj_X = tf.linalg.det(x)
    # else:
    #     adj_X = tf.linalg.inv(x) * tf.linalg.det(x)
    return adj_X