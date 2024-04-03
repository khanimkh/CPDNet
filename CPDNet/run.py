import argparse
import subprocess
import tensorflow as tf
import numpy as np
from datetime import datetime
import json
import os
import sys
import datetime
import time
import collections


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import P2PNET
import ioUtil


# DEFAULT SETTINGS
parser = argparse.ArgumentParser()

parser.add_argument('--train_hdf5', default='data_hdf5/spine_train.hdf5' )
parser.add_argument('--test_hdf5', default='data_hdf5/spine_test.hdf5' )

parser.add_argument('--domain_A', default='skeleton', help='name of domain A')
parser.add_argument('--domain_B', default='surface',  help='name of domain B')
parser.add_argument('--meta_A', default='meta',  help='metadata of domain A')
parser.add_argument('--pre_T', default='pre_T',  help='translation data of domain A')
parser.add_argument('--pre_R', default='pre_R',  help='rotatdion data of domain A')
parser.add_argument('--post_T', default='post_T',  help='translation data of domain B')
parser.add_argument('--post_R', default='post_R',  help='rotation data of domain B')

parser.add_argument('--mode', type=str, default='train', help='train or test')
parser.add_argument('--modeCode', type=str, default='create', help='train or test')
parser.add_argument('--gpu', type=int, default=0, help='which GPU to use [default: 0]')
parser.add_argument('--batch_size', type=int, default=6, help='Batch Size during training [default: 4]')
parser.add_argument('--epoch', type=int, default=500, help='number of epoches to run [default: 200]')
parser.add_argument('--decayEpoch',  type=int, default=200, help='steps(how many epoches) for decaying learning rate')
parser.add_argument('--learning_rate',  type=float, default=0.005, help='learning rate')

parser.add_argument("--shapeWeight", type=float, default=1, help="density weight [default: 1.0]")
parser.add_argument("--densityWeight", type=float, default=0, help="density weight [default: 1.0]")
parser.add_argument("--regularWeight", type=float, default=0, help="regularization weight [default: 0.1]")
parser.add_argument("--localWeight", type=float, default=0, help="regularization weight [default: 1.0]")
parser.add_argument("--metaWeight", type=float, default=2, help="regularization weight [default: 1.0]")
parser.add_argument("--nnk", type=int, default=24, help="density:  number of nearest neighbours [default: 8]")

parser.add_argument("--range_max", type=float, default=150, help="max length of point displacement[default: 1.0]")
parser.add_argument("--radiusScal", type=float, default=1, help="a constant for scaling radii in pointnet++ [default: 1.0]")
parser.add_argument("--noiseLength", type=int, default=20, help="length of point-wise noise vector [default: 32]")
parser.add_argument("--metaLength", type=int, default=6, help="length of point-wise noise vector [default: 4]")

parser.add_argument('--checkpoint', default=None, help='output_spine_skeleton-surface/trained_models/epoch_400.ckpt')

###  None  None  None
parser.add_argument('--point_num', type=int, default=108, help='default 108')
parser.add_argument('--point_num_meta', type=int, default=108, help='default 108')
parser.add_argument('--vertebra_num', type=int, default=102, help='do not set the argument')
parser.add_argument('--example_num', type=int, default=None, help='do not set the argument')
parser.add_argument('--output_dir', type=str,  default=None, help='do not set the argument')

FLAGS = parser.parse_args()

Train_examples = ioUtil.load_examples(FLAGS.train_hdf5, FLAGS.domain_A, FLAGS.domain_B, FLAGS.meta_A, FLAGS.pre_T, FLAGS.pre_R, FLAGS.post_T, FLAGS.post_R, 'names')
Eval_examples  = ioUtil.load_eval_examples(FLAGS.test_hdf5,  FLAGS.domain_A, FLAGS.domain_B, FLAGS.meta_A, FLAGS.pre_T, FLAGS.pre_R, FLAGS.post_T, FLAGS.post_R, 'names')
Test_examples  = ioUtil.load_test_examples(FLAGS.test_hdf5,  FLAGS.domain_A, FLAGS.domain_B, FLAGS.meta_A, FLAGS.pre_T, FLAGS.pre_R, FLAGS.post_T, FLAGS.post_R, 'names')


FLAGS.point_num = Train_examples.pointsets_A.shape[1]
FLAGS.vertebra_num = Train_examples.pointsets_T_A.shape[1]
POINT_NUM = FLAGS.point_num

Example_NUM = Train_examples.pointsets_A.shape[0]
FLAGS.example_num =  Example_NUM

TRAINING_EPOCHES = FLAGS.epoch

batch_size = FLAGS.batch_size

if Train_examples.pointsets_B.shape[1] != POINT_NUM \
    or Eval_examples.pointsets_A.shape[1] != POINT_NUM \
    or Eval_examples.pointsets_B.shape[1] != POINT_NUM :
    print( 'point number inconsistent in the data set.')
    exit()

########## create output folders
datapath, basefname = os.path.split( FLAGS.train_hdf5 )
output_dir = 'output_' + basefname[0:basefname.index('_')] + '_' + FLAGS.domain_A + '-' + FLAGS.domain_B ## + '_noise' + str(FLAGS.noiseLength) + '_dw' + str(FLAGS.densityWeight)+ '_rw' + str(FLAGS.regularWeight)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

MODEL_STORAGE_PATH = os.path.join(output_dir, 'trained_models')
if not os.path.exists(MODEL_STORAGE_PATH):
    os.mkdir(MODEL_STORAGE_PATH)

SUMMARIES_FOLDER = os.path.join(output_dir, 'summaries')
if not os.path.exists(SUMMARIES_FOLDER):
    os.mkdir(SUMMARIES_FOLDER)


########## Save test input
# ioUtil.output_point_cloud_ply( Eval_examples.pointsets_A, Eval_examples.names, output_dir, 'gt_'+FLAGS.domain_A)
# ioUtil.output_point_cloud_ply( Eval_examples.pointsets_B, Eval_examples.names, output_dir, 'gt_'+FLAGS.domain_B)

# print arguments
for k, v in FLAGS._get_kwargs():
    print(k + ' = ' + str(v) )


def train():
    with tf.Graph().as_default() as g:
        with tf.device('/cpu:' + str(FLAGS.gpu)):
            model = P2PNET.create_model(FLAGS)


        ########## Init and Configuration   ##########
        saver = tf.train.Saver( max_to_keep=25 )
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        ### add by maryam
        config.gpu_options.per_process_gpu_memory_fraction = 0.90
        config.log_device_placement=True
        sess = tf.Session(config=config)
        #sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
        #sess= tf.ConfigProto(device_count={'GPU': 2, 'CPU': 2}, allow_soft_placement=True)

        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        sess.run(init_g)
        sess.run(init_l)
        #init = tf.global_variables_initializer()
        #sess.run(init)

        # Restore variables from disk.
        Start_epoch_number = 1
        if FLAGS.checkpoint is not None:
            print('load checkpoint: ' + FLAGS.checkpoint)
            saver.restore(sess, FLAGS.checkpoint )

            fname = os.path.basename( FLAGS.checkpoint )
            Start_epoch_number = int( fname[6:-5] )  +  1

            print( 'Start_epoch_number = ' + str(Start_epoch_number) )

        now = datetime.datetime.now()
        var_date= now.strftime("%Y-%m-%d %H:%M")

        train_writer = tf.summary.FileWriter("%s/train_%.4f_%.4f_%.4f_%d_%d_%.4f_%d_%s" % (SUMMARIES_FOLDER, FLAGS.localWeight, FLAGS.regularWeight, FLAGS.densityWeight, FLAGS.batch_size,FLAGS.radiusScal, FLAGS.learning_rate,FLAGS.metaWeight,var_date) , sess.graph)
        #train_writer = tf.summary.FileWriter(SUMMARIES_FOLDER + '/train', sess.graph)

        test_writer = tf.summary.FileWriter("%s/test_%.4f_%.4f_%.4f_%d_%d_%.4f_%d_%s" % (SUMMARIES_FOLDER, FLAGS.localWeight, FLAGS.regularWeight, FLAGS.densityWeight, FLAGS.batch_size,FLAGS.radiusScal, FLAGS.learning_rate,FLAGS.metaWeight,var_date))
        #test_writer = tf.summary.FileWriter(SUMMARIES_FOLDER + '/test')

        fcmd = open(os.path.join(output_dir, 'arguments.txt'), 'w')
        fcmd.write(str(FLAGS))
        fcmd.close()


        ########## Training one epoch  ##########

        def train_one_epoch(epoch_num):

            now = datetime.datetime.now()
            print(now.strftime("%Y-%m-%d %H:%M:%S"))
            start_time = time.time()

            is_training = True

            Train_examples_shuffled = ioUtil.shuffle_examples(Train_examples)

            pointsets_A = Train_examples_shuffled.pointsets_A
            pointsets_B = Train_examples_shuffled.pointsets_B
            pointsets_meta_A = Train_examples_shuffled.pointsets_meta_A
            pointsets_T_A = Train_examples_shuffled.pointsets_T_A
            pointsets_R_A = Train_examples_shuffled.pointsets_R_A
            pointsets_T_B = Train_examples_shuffled.pointsets_T_B
            pointsets_R_B = Train_examples_shuffled.pointsets_R_B
            names = Train_examples_shuffled.names

            num_data = pointsets_A.shape[0]
            num_batch = num_data // batch_size

            total_total_loss = 0.0
            total_data_loss_A = 0.0
            total_shape_loss_A = 0.0
            total_density_loss_A = 0.0


            total_data_loss_B = 0.0
            total_shape_loss_B = 0.0
            total_density_loss_B = 0.0

            total_rigid_loss_A = 0.0
            total_rigid_loss_B = 0.0

            total_reg_loss = 0.0

            for j in range(num_batch):

                begidx = j * batch_size
                endidx = (j + 1) * batch_size

                feed_dict = {
                    model.pointSet_A_ph: pointsets_A[begidx: endidx, ...],
                    model.pointSet_B_ph: pointsets_B[begidx: endidx, ...],
                    model.pointSet_meta_A_ph: pointsets_meta_A[begidx: endidx, ...],
                    model.pointSet_T_A_ph: pointsets_T_A[begidx: endidx, ...],
                    model.pointSet_R_A_ph: pointsets_R_A[begidx: endidx, ...],
                    model.pointSet_T_B_ph: pointsets_T_B[begidx: endidx, ...],
                    model.pointSet_R_B_ph: pointsets_R_B[begidx: endidx, ...],
                    model.is_training_ph: is_training,
                }

                fetches = {
                    "train": model.total_train,
                    "shapeLoss_A": model.shapeLoss_A,
                    "densityLoss_A": model.densityLoss_A,
                    "shapeLoss_B": model.shapeLoss_B,
                    "densityLoss_B": model.densityLoss_B,
                    "total_loss": model.total_loss,
                    "data_loss_A": model.data_loss_A,
                    "data_loss_B": model.data_loss_B,
                    "rigid_loss_A": model.rigid_loss_A,
                    "rigid_loss_B": model.rigid_loss_B,
                    "regul_loss": model.regul_loss,
                    "learning_rate": model.learning_rate,
                    "global_step": model.global_step,
                    "Predicted_A": model.Predicted_A,
                    "Predicted_B": model.Predicted_B,
                    "displace_B2A":model.list_dis_B2A_all,
                    "displace_A2B": model.list_dis_A2B_all,
                    "A":model.pointSet_A_ph,
                    "B":model.pointSet_B_ph,
                }


                results = sess.run(fetches, feed_dict=feed_dict)
                A=results["A"]
                B=results["B"]
                Predicted_A = results["Predicted_A"]
                Predicted_B = results["Predicted_B"]
                displace_B2A = results["displace_B2A"]
                displace_A2B = results["displace_A2B"]
                # dis_A2B =  results["dis_A2B"]
                # dis_B2A =  results["dis_B2A"]

                total_total_loss += results["total_loss"]
                total_data_loss_A += results["data_loss_A"]
                total_shape_loss_A += results["shapeLoss_A"]
                total_density_loss_A += results["densityLoss_A"]


                total_data_loss_B += results["data_loss_B"]
                total_shape_loss_B += results["shapeLoss_B"]
                total_density_loss_B += results["densityLoss_B"]

                total_rigid_loss_A += results["rigid_loss_A"]
                total_rigid_loss_B += results["rigid_loss_B"]

                total_reg_loss += results["regul_loss"]



                # if j % 1 == 0:
                #     print('    ' + str(j) + '/' + str(num_batch) + ':    '  )
                #     print('            data_loss_A = {:.4f},'.format(results["data_loss_A"] )  +  \
                #           '   shape = {:.4f},'.format(results["shapeLoss_A"] )           + \
                #           '   density = {:.4f}'.format(results["densityLoss_A"] )   )
                #
                #     print('            data_loss_B = {:.4f},'.format(results["data_loss_B"] )  + \
                #           '   shape = {:.4f},'.format(results["shapeLoss_B"] )           + \
                #           '   density = {:.4f}'.format(results["densityLoss_B"] )   )
                #
                #     print('            rigid_loss_A = {:.4f}\n'.format(results["rigid_loss_A"]))
                #
                #     print('            rigid_loss_B = {:.4f}\n'.format(results["rigid_loss_B"]))
                #
                #     print('            regul_loss = {:.4f}\n'.format(results["regul_loss"] ) )
                #
                #     print('            learning_rate = {:.6f}'.format(results["learning_rate"] )  )
                #     print('            global_step = {0}'.format(results["global_step"] )  )

            total_total_loss /= num_batch
            total_data_loss_A      /= num_batch
            total_shape_loss_A  /= num_batch
            total_density_loss_A   /= num_batch
            total_data_loss_B      /= num_batch
            total_shape_loss_B  /= num_batch
            total_density_loss_B   /= num_batch
            total_rigid_loss_A /= num_batch
            total_rigid_loss_B /= num_batch
            total_reg_loss         /= num_batch

            # evaluate summaries
            training_sum = sess.run( model.training_sum_ops, \
                                    feed_dict={model.train_dataloss_A_ph: total_data_loss_A, \
                                               model.train_dataloss_B_ph: total_data_loss_B, \
                                               model.train_totalloss_ph: total_total_loss, \
                                               model.train_rigidloss_A_ph: total_rigid_loss_A, \
                                               model.train_rigidloss_B_ph: total_rigid_loss_B, \
                                               model.train_regul_ph: total_reg_loss, \
                                               })

            train_writer.add_summary(training_sum, epoch_num)


            print(  '\tData_loss_A = %.4f,' % total_data_loss_A    + \
                    '    shape = %.4f,' % total_shape_loss_A + \
                    '    density = %.4f' % total_density_loss_A )

            print(  '\tData_loss_B = %.4f,' % total_data_loss_B    + \
                    '    shape = %.4f,' % total_shape_loss_B + \
                    '    density = %.4f' % total_density_loss_B )

            print(  '\tReg_loss: %.4f\n' % total_reg_loss)

            print('\tRigid_loss_A: %.4f\n' % total_rigid_loss_A)

            print('\tRigid_loss_B: %.4f\n' % total_rigid_loss_B)


            elapsed_time = time.time() - start_time
            print( '\tply/sec:' + str( round(num_data/elapsed_time) ) )
            print( '\tduration of this epoch:' + str(round(elapsed_time/60) ) + ' min' )
            print( '\testimated finishing time:' + str(round(elapsed_time/60.0 * (TRAINING_EPOCHES-epoch_num-1)) ) + ' min' )

        ################## end  of train function #################### end  of train function ##########


        def eval_one_epoch(epoch_num, mustSavePly=False):
            is_training = False
            Eval_examples_shuffled = ioUtil.shuffle_examples(Eval_examples)
            Test_examples_shuffled = ioUtil.shuffle_examples(Test_examples)

            if FLAGS.mode=='train':
              pointsets_A = Eval_examples.pointsets_A
              pointsets_B = Eval_examples.pointsets_B
              pointsets_meta_A = Eval_examples.pointsets_meta_A
              pointsets_T_A = Eval_examples.pointsets_T_A
              pointsets_R_A = Eval_examples.pointsets_R_A
              pointsets_T_B = Eval_examples.pointsets_T_B
              pointsets_R_B = Eval_examples.pointsets_R_B
              names = Eval_examples.names
            else:
              pointsets_A = Test_examples.pointsets_A
              pointsets_B = Test_examples.pointsets_B
              pointsets_meta_A = Test_examples.pointsets_meta_A
              pointsets_T_A = Test_examples.pointsets_T_A
              pointsets_R_A = Test_examples.pointsets_R_A
              pointsets_T_B = Test_examples.pointsets_T_B
              pointsets_R_B = Test_examples.pointsets_R_B
              names = Test_examples.names

            num_data = pointsets_A.shape[0]
            num_batch = num_data // batch_size

            total_total_loss = 0.0
            total_data_loss_A = 0.0
            total_shape_loss_A = 0.0
            total_density_loss_A = 0.0

            total_data_loss_B = 0.0
            total_shape_loss_B = 0.0
            total_density_loss_B = 0.0

            total_rigid_loss_A = 0.0
            total_rigid_loss_B = 0.0

            total_reg_loss = 0.0

            for j in range(num_batch):

                begidx = j * batch_size
                endidx = (j + 1) * batch_size

                feed_dict = {
                    model.pointSet_A_ph: pointsets_A[begidx: endidx, ...],
                    model.pointSet_B_ph: pointsets_B[begidx: endidx, ...],
                    model.pointSet_meta_A_ph: pointsets_meta_A[begidx: endidx, ...],
                    model.pointSet_T_A_ph: pointsets_T_A[begidx: endidx, ...],
                    model.pointSet_R_A_ph: pointsets_R_A[begidx: endidx, ...],
                    model.pointSet_T_B_ph: pointsets_T_B[begidx: endidx, ...],
                    model.pointSet_R_B_ph: pointsets_R_B[begidx: endidx, ...],
                    model.is_training_ph: is_training,
                }

                fetches = {
                    "shapeLoss_A": model.shapeLoss_A,
                    "densityLoss_A": model.densityLoss_A,
                    "shapeLoss_B": model.shapeLoss_B,
                    "densityLoss_B": model.densityLoss_B,
                    "total_loss": model.total_loss,
                    "data_loss_A": model.data_loss_A,
                    "data_loss_B": model.data_loss_B,
                    "rigid_loss_A": model.rigid_loss_A,
                    "rigid_loss_B": model.rigid_loss_B,
                    "regul_loss": model.regul_loss,
                    "Predicted_A": model.Predicted_A,
                    "Predicted_B": model.Predicted_B,
                    "displace_B2A": model.list_dis_B2A_all,
                    "displace_A2B": model.list_dis_A2B_all,
                    "Accuracy": model.Accuracy,
                }


                results = sess.run(fetches, feed_dict=feed_dict)

                total_total_loss += results["total_loss"]
                total_data_loss_A += results["data_loss_A"]
                total_shape_loss_A += results["shapeLoss_A"]
                total_density_loss_A += results["densityLoss_A"]

                total_data_loss_B += results["data_loss_B"]
                total_shape_loss_B += results["shapeLoss_B"]
                total_density_loss_B += results["densityLoss_B"]

                total_rigid_loss_A += results["rigid_loss_A"]
                total_rigid_loss_B += results["rigid_loss_B"]

                total_reg_loss += results["regul_loss"]
                Accuracy = results["Accuracy"]


                # write test results
                if FLAGS.mode=='test' and (epoch_num == 200 or epoch_num == 300  or epoch_num == 400 or epoch_num == 500 or epoch_num == 600 or epoch_num == 700 or  epoch_num == 800 or epoch_num == 900 or epoch_num == 1000 or mustSavePly):

                    # save predicted point sets with 1 single feeding pass
                    nametosave = names[begidx: endidx, ...]
                    Predicted_A_xyz = np.squeeze(np.array(results["Predicted_A"]))
                    Predicted_B_xyz = np.squeeze(np.array(results["Predicted_B"]))

                    # file_name_test_A=str('%s/test_%.4f_%.4f_%.4f_%d_%d_%.4f_%s' % (SUMMARIES_FOLDER, FLAGS.localWeight, FLAGS.regularWeight, FLAGS.densityWeight, FLAGS.batch_size,
                    # FLAGS.radiusScal, FLAGS.learning_rate, var_date)) +'Ep' + str(epoch_num) + '_predicted_' + FLAGS.domain_A + 'X1'
                    #
                    # file_name_test_B = str('%s/test_%.4f_%.4f_%.4f_%d_%d_%.4f_%s' % (SUMMARIES_FOLDER, FLAGS.localWeight, FLAGS.regularWeight, FLAGS.densityWeight, FLAGS.batch_size,
                    # FLAGS.radiusScal, FLAGS.learning_rate, var_date)) + 'Ep' + str(epoch_num) + '_predicted_' + FLAGS.domain_B + 'X1'

                    # ioUtil.output_point_cloud_ply(Predicted_A_xyz, nametosave, output_dir, file_name_test_A)
                    # ioUtil.output_point_cloud_ply(Predicted_B_xyz, nametosave, output_dir, file_name_test_B)

                    ioUtil.output_point_cloud_ply(Predicted_A_xyz, nametosave, output_dir,
                                                  'Ep' + str(epoch_num) + '_predicted_' + FLAGS.domain_A + 'X1')
                    ioUtil.output_point_cloud_ply(Predicted_B_xyz, nametosave, output_dir,
                                                  'Ep' + str(epoch_num) + '_predicted_' + FLAGS.domain_B + 'X1')

                    # save predicted point sets with 4 feeding passes
                    # for i in range(3):
                    #    results = sess.run(fetches, feed_dict=feed_dict)
                    #    Predicted_A_xyz__ = np.squeeze(np.array(results["Predicted_A"]))
                    #    Predicted_B_xyz__ = np.squeeze(np.array(results["Predicted_B"]))
                    #    Predicted_A_xyz = np.concatenate((Predicted_A_xyz, Predicted_A_xyz__), axis=1)
                    #    Predicted_B_xyz = np.concatenate((Predicted_B_xyz, Predicted_B_xyz__), axis=1)
                    # displace_B2A = results["displace_B2A"]
                    # displace_A2B = results["displace_A2B"]
                    #
                    # ioUtil.output_point_cloud_ply(Predicted_A_xyz + displace_B2A, nametosave, output_dir,
                    #                                'Ep' + str(epoch_num) + '_predicted_' + FLAGS.domain_A + 'X2')
                    # ioUtil.output_point_cloud_ply(Predicted_B_xyz + displace_A2B, nametosave, output_dir,
                    #                                'Ep' + str(epoch_num) + '_predicted_' + FLAGS.domain_B + 'X2')

                    # save predicted point sets with 8 feeding passes
                    # for i in range(4):
                    #    results = sess.run(fetches, feed_dict=feed_dict)
                    #    Predicted_A_xyz__ = np.squeeze(np.array(results["Predicted_A"]))
                    #    Predicted_B_xyz__ = np.squeeze(np.array(results["Predicted_B"]))
                    #    Predicted_A_xyz = np.concatenate((Predicted_A_xyz, Predicted_A_xyz__), axis=1)
                    #    Predicted_B_xyz = np.concatenate((Predicted_B_xyz, Predicted_B_xyz__), axis=1)
                    #
                    # ioUtil.output_point_cloud_ply( Predicted_A_xyz, nametosave, output_dir,
                    #                                'Ep' + str(epoch_num) + '_predicted_' + FLAGS.domain_A + 'X8')
                    # ioUtil.output_point_cloud_ply( Predicted_B_xyz, nametosave, output_dir,
                    #                                'Ep' + str(epoch_num) + '_predicted_' + FLAGS.domain_B + 'X8')

            total_total_loss /= num_batch
            total_data_loss_A      /= num_batch
            total_shape_loss_A  /= num_batch
            total_density_loss_A   /= num_batch
            total_data_loss_B      /= num_batch
            total_shape_loss_B  /= num_batch
            total_density_loss_B   /= num_batch
            total_rigid_loss_A /= num_batch
            total_rigid_loss_B /= num_batch
            total_reg_loss         /= num_batch

            # evaluate summaries
            testing_sum = sess.run( model.testing_sum_ops, \
                                    feed_dict={model.test_dataloss_A_ph: total_data_loss_A, \
                                               model.test_dataloss_B_ph: total_data_loss_B, \
                                               model.test_totalloss_ph: total_total_loss, \
                                               model.test_regul_ph: total_reg_loss, \
                                               model.test_rigidloss_A_ph: total_rigid_loss_A, \
                                               model.test_rigidloss_B_ph: total_rigid_loss_B, \
                                               })

            test_writer.add_summary(testing_sum, epoch_num)


            print('\tData_loss_A = %.4f,' % total_data_loss_A  + \
                  '    shape = %.4f,' % total_shape_loss_A + \
                  '    density = %.4f' % total_density_loss_A)

            print('\tData_loss_B = %.4f,' % total_data_loss_B + \
                  '    shape = %.4f,' % total_shape_loss_B + \
                  '    density = %.4f' % total_density_loss_B)

            print('\tRigid_loss_A: %.4f\n' % total_rigid_loss_A)

            print('\tRigid_loss_B: %.4f\n' % total_rigid_loss_B)

            print('\tReg_loss: %.4f\n' % total_reg_loss)

            print('\tAccuracy: %.4f\n' % Accuracy)

        ################## end  of test function #################### end  of test function ##########

        if not os.path.exists(MODEL_STORAGE_PATH):
            os.mkdir(MODEL_STORAGE_PATH)

        if FLAGS.mode=='train':
            for epoch in range(Start_epoch_number,  TRAINING_EPOCHES+1):

                print( '\n>>> Training for the epoch %d/%d ...' % (epoch, TRAINING_EPOCHES))
                train_one_epoch(epoch)

                if epoch % 20 == 0:

                    cp_filename = saver.save(sess, os.path.join(MODEL_STORAGE_PATH, 'epoch_' + str(epoch) + '.ckpt'))
                    print( 'Successfully store the checkpoint model into ' + cp_filename)

                    print('\n<<< Testing on the test dataset...')
                    eval_one_epoch(epoch, mustSavePly=False)

        else:

            print( '\n<<< Testing on the test dataset ...')
            eval_one_epoch(Start_epoch_number, mustSavePly=True)


    train_writer.close()
    test_writer.close()
if __name__ == '__main__':
    train()
