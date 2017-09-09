# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import regularizers
import numpy as np
import random
from scipy import ndimage
import time
import os
import sys
from guotai_brats.data_process.data_loader import *
from guotai_brats.data_process.pre_process import save_array_as_nifty_volume
from guotai_brats.util.train_test_func import *
from guotai_brats.util.parse_config import parse_config
from guotai_brats.mylayer.crfrnnf3d import PairwisePotentialFunctionLayer
from layer.loss import LossFunction
from guotai_run import NetFactory
from guotai_nine_stage_test import get_roi

        
def get_largest_two_component(img, prt = False, threshold = None):
    s = ndimage.generate_binary_structure(3,2) # iterate structure
    labeled_array, numpatches = ndimage.label(img,s) # labeling
    sizes = ndimage.sum(img,labeled_array,range(1,numpatches+1)) 
    sizes_list = [sizes[i] for i in range(len(sizes))]
    sizes_list.sort()
    if(prt):
        print("component size", sizes_list)
    if(len(sizes) == 1):
        return img
    else:
        if(threshold):
            out_img = np.zeros_like(img)
            for temp_size in sizes_list:
                if(temp_size > threshold):
                    temp_lab = np.where(sizes == temp_size)[0] + 1
                    temp_cmp = labeled_array == temp_lab
                    out_img = (out_img + temp_cmp) > 0
            return out_img
        else:    
            max_size1 = sizes_list[-1]
            max_size2 = sizes_list[-2]
            max_label1 = np.where(sizes == max_size1)[0] + 1
            max_label2 = np.where(sizes == max_size2)[0] + 1
            component1 = labeled_array == max_label1
            component2 = labeled_array == max_label2
            if(prt):
                print(max_size2, max_size1, max_size2/max_size1)   
            if(max_size2*10 > max_size1):
                component1 = (component1 + component2) > 0
            
            return component1
def run(stage, config_file):
    # construct graph
    config = parse_config(config_file)
    config_data  = config['data']
    config_net   = config['network']
    config_train = config['training']
    config_test  = config['testing']  
    random.seed(config_train.get('random_seed', 1)) 
    if(stage == 'train'):
        assert(config_data['with_ground_truth'])

    net_type    = config_net['net_type']
    net_name    = config_net['net_name']
    data_shape  = config_net['data_shape']
    label_shape = config_net['label_shape']
    data_channel= config_net['data_channel']
    class_num   = config_net['class_num']
    batch_size  = config_data.get('batch_size', 5)
   
    # construct graph
    full_data_shape = [batch_size] + data_shape + [data_channel]
    full_label_shape = [batch_size] + label_shape + [1]
    x = tf.placeholder(tf.float32, shape = full_data_shape)
    w = tf.placeholder(tf.float32, shape = full_label_shape)
    y = tf.placeholder(tf.int64, shape = full_label_shape)  
   
    w_regularizer = None; b_regularizer = None
    if(stage == 'train'):
        w_regularizer = regularizers.l2_regularizer(config_train.get('decay', 1e-7))
        b_regularizer = regularizers.l2_regularizer(config_train.get('decay', 1e-7))
    net_class = NetFactory.create(net_type)
    if(net_type == 'DeepMedic'):
        net = net_class(10, 57, 9, 2, True, "gpu")
    else:
        net = net_class(num_classes = class_num,
                    w_regularizer = w_regularizer,
                    b_regularizer = b_regularizer,
                    name = net_name)
    if hasattr(net, 'set_params'):
        net.set_params(config_net)
    
    loss_func = LossFunction(n_class=class_num)
    alpha = tf.placeholder(tf.float32, name = 'alpha')
  
    if(net_type == 'CRFRNNF3DLayer' ):
        full_prob_shape = [batch_size] + data_shape + [2]
        p = tf.placeholder(tf.float32, shape = full_prob_shape)
        predicty= net([x, p], is_training = stage == 'train')
        
        # graph for pairwise function
        pair_batch_size = [100, 1, 10, 10]
        pair_x = tf.placeholder(tf.float32, shape = pair_batch_size + [2])
        potential_func = PairwisePotentialFunctionLayer([32, 16], name = 'pairwise_net')
        pair_y = potential_func(pair_x)  
    elif(net_type == 'DeepMedic'):
        predicty = net.inference(x, None)  
    else:
        predicty = net(x, is_training = True)
    proby = tf.nn.softmax(predicty)
    loss = loss_func(predicty, y, weight_map = w)
    print('size of predicty:',predicty)
    
    # Initialize session and saver
    if(stage == 'train'):
        lr = config_train.get('learning_rate', 1e-3)
        opt_step = tf.train.AdamOptimizer(lr).minimize(loss)
        
    all_vars = tf.global_variables()
    sess = tf.InteractiveSession()   
    sess.run(tf.global_variables_initializer())  
    saver = tf.train.Saver()
    
    # load pre-trained pairwise potential
    if(net_type == 'CRFRNNF3DLayer' and stage == 'train'):
        pairwise_var = [xi for xi in all_vars if 'pairwise_function' in xi.name]
        pairwise_pretrain_var =  [xi for xi in all_vars if 'pairwise_net' in xi.name]
#         pair_model_file = config_net.get('pair_model', None)
#         pair_saver = tf.train.Saver(pairwise_pretrain_var)
#         pair_saver.restore(sess, pair_model_file)
#         for i in range(len(pairwise_pretrain_var)):
#             copy_value = tf.assign(pairwise_var[i], pairwise_pretrain_var[i])
#             print(pairwise_var[i].name, pairwise_pretrain_var[i].name)
#             print(copy_value.eval())
#             print(pairwise_var[i].eval())
                 
    loader = BratsLoader()
    loader.set_params(config_data)
    loader.load_data()

    if(stage == 'train'):
        loss_list = [] 
        loss_file = config_train['model_save_prefix'] + "_loss.txt"
        start_it = config_train.get('start_iteration', 0)
        if( start_it> 0):
            saver.restore(sess, config_train['model_pre_trained'])
        for n in range(start_it, config_train['maximal_iteration']):
            train_pair = loader.get_subimage_batch()
            tempx = train_pair['images']
            tempw = train_pair['weights']
            tempy = train_pair['labels']
            tempp = train_pair['probs']
            alpha_value = (n + 0.0)/config_train['maximal_iteration']
            if(net_type == 'CRFRNNF3DLayer'):
                tempp = np.concatenate((1.0 - tempp, tempp), -1)
                opt_step.run(session = sess, feed_dict={x:tempx, w: tempw, y:tempy, p:tempp})
            else:
                opt_step.run(session = sess, feed_dict={x:tempx, w: tempw, y:tempy, alpha: alpha_value})
              
            if(n%config_train['test_iteration'] == 0):
                batch_dice_list = []
                for step in range(config_train['test_step']):
                    train_pair = loader.get_subimage_batch()
                    tempx = train_pair['images']
                    tempw = train_pair['weights']
                    tempy = train_pair['labels']
                    tempp = train_pair['probs']
                    if(net_type == 'CRFRNNF3DLayer'):
                        tempp = np.concatenate((1.0 - tempp, tempp), -1)
                        dice = loss.eval(feed_dict ={x:tempx, w:tempw, y:tempy , p: tempp})
                    else:
                        dice = loss.eval(feed_dict ={x:tempx, w:tempw, y:tempy , alpha: alpha_value})
                    batch_dice_list.append(dice)
                batch_dice = np.asarray(batch_dice_list, np.float32).mean()
                t = time.strftime('%X %x %Z')
                print(t, 'n', n,'loss', batch_dice)
                loss_list.append(batch_dice)
                np.savetxt(loss_file, np.asarray(loss_list))
            if((n+1)%config_train['snapshot_iteration']  == 0):
                saver.save(sess, config_train['model_save_prefix']+"_{0:}.ckpt".format(n+1))
    else:
        saver.restore(sess, config_test['model_file'])
        test_slice_direction = config_test.get('test_slice_direction', 'axial')
        save_folder = config_test['save_folder']
        test_with_roi = config_test.get('test_with_roi', False)
        down_sample = config_test.get('down_sample_rate', 1.0)
        data_resize = config_test.get('data_resize', None)
        label_convert_source = config_test.get('label_convert_source', None)
        label_convert_target = config_test.get('label_convert_target', None)
        save_prob = config_test.get('save_prob', False)
        image_num = loader.get_total_image_number()
        test_time = []
        for i in range(image_num):
            if(net_type == 'CRFRNNF3DLayer'):
                [load_imgs, load_weight, load_prob, load_name] = loader.get_image_data_with_name(i, True)
            else:
                [load_imgs, load_weight, load_name] = loader.get_image_data_with_name(i)
            print(i, 'out of ', image_num, load_name)
         
            if(test_with_roi):
                if(net_type == 'CRFRNNF3DLayer'):
                    groi = get_roi(load_prob > 0.5, 5)
                    roi_prob = load_prob[np.ix_(range(groi[0], groi[1]), range(groi[2], groi[3]), range(groi[4], groi[5]))]
                else:
                    groi = get_roi(load_weight > 0, 5)
                roi_imgs = [x[np.ix_(range(groi[0], groi[1]), range(groi[2], groi[3]), range(groi[4], groi[5]))] \
                            for x in load_imgs]
                roi_weight = load_weight[np.ix_(range(groi[0], groi[1]), range(groi[2], groi[3]), range(groi[4], groi[5]))]
            else:
                roi_imgs = load_imgs
                roi_weight = load_weight
                if(net_type == 'CRFRNNF3DLayer'):
                    roi_prob = load_prob
                    roi_prob = np.asarray([1.0 - roi_prob, roi_prob])
                    roi_prob = np.transpose(roi_prob, [1, 2, 3, 0])    
                
            if(down_sample == 1.0):
                down_imgs = roi_imgs
                down_weight = roi_weight
            else:
                down_imgs = []
                for mod in range(len(roi_imgs)):
                    down_imgs.append(ndimage.interpolation.zoom(roi_imgs[mod],1.0/down_sample, order = 1))
                down_weight = ndimage.interpolation.zoom(roi_weight,1.0/down_sample, order = 1)
                
            if(data_resize):
                temp_imgs = []
                for mod in range(len(down_imgs)):
                    temp_imgs.append(resize_3D_volume_to_given_shape(down_imgs[mod], data_resize, order = 1))
                temp_weight = resize_3D_volume_to_given_shape(down_weight, data_resize, order = 1)
            else:
                temp_imgs = down_imgs
                temp_weight = down_weight
            t0 = time.time()

            if(net_type == 'CRFRNNF3DLayer'):
                temp_prob =  volume_crf_probability_prediction_3d_roi(temp_imgs, roi_prob, data_shape, label_shape, data_channel, 
                      class_num, batch_size, sess, proby, x, p)
            elif(test_with_roi):
                temp_prob = volume_probability_prediction_3d_roi(temp_imgs, data_shape, label_shape, data_channel, 
                                      class_num, batch_size, sess, proby, x)
            else:
                temp_prob = test_one_image(temp_imgs, data_shape, label_shape, data_channel, class_num,
                                                         batch_size, test_slice_direction, sess, proby, x)

            if(down_sample != 1.0 or data_resize):
                temp_prob = resize_ND_volume_to_given_shape(temp_prob, list(roi_weight.shape) + [class_num], order = 1)
            temp_time = time.time() - t0
            test_time.append(temp_time)
            temp_label =  np.argmax(temp_prob, axis = 3)
            temp_label[roi_weight==0] = 0
#             temp_label = get_largest_two_component(temp_label) # use this for real user interactions
            temp_label = np.asarray(temp_label, np.uint16)

            if(label_convert_source and label_convert_target):
                assert(len(label_convert_source) == len(label_convert_target))
                temp_label = convert_label(label_convert_source, label_convert_target)
            
            if(test_with_roi):
                final_label = np.zeros_like(load_weight, np.int16)
                final_label[np.ix_(range(groi[0], groi[1]), range(groi[2], groi[3]), range(groi[4], groi[5]))] = temp_label
            else:
                final_label = temp_label
            save_array_as_nifty_volume(final_label, save_folder+"/{0:}.nii.gz".format(load_name))

            if(save_prob):
                fg_prob = temp_prob[:,:,:,1]
                fg_prob = np.reshape(fg_prob, final_label.shape)
                save_array_as_nifty_volume(fg_prob, save_folder+"_prob/{0:}.nii.gz".format(load_name))
        test_time = np.asarray(test_time)
        print('test time', test_time.mean(), test_time.std())
        np.savetxt(save_folder + '/test_time.txt', test_time)
    
    sess.close()
    
if __name__ == '__main__':
    if(len(sys.argv) != 3):
        print('Number of arguments should be 3. e.g.')
        print('    python guotai_run.py train config.txt')
        exit()
    stage = str(sys.argv[1])
    config_file = str(sys.argv[2])
    assert(os.path.isfile(config_file))
    run(stage, config_file)
    
    