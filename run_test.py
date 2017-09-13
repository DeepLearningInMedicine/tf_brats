# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import tensorflow as tf
import numpy as np
from scipy import ndimage
import time
import os
import sys
from data_io.data_loader import *
from run_train import NetFactory
from util.train_test_func import test_one_image_three_nets, test_one_image_three_nets_adaptive_shape
from util.parse_config import parse_config
import pickle

 
def get_roi(temp_label, margin):
    [d_idxes, h_idxes, w_idxes] = np.nonzero(temp_label)
    [D, H, W] = temp_label.shape
    mind = max(d_idxes.min() - margin, 0)
    maxd = min(d_idxes.max() + margin, D)
    minh = max(h_idxes.min() - margin, 0)
    maxh = min(h_idxes.max() + margin, H)
    minw = max(w_idxes.min() - margin, 0)
    maxw = min(w_idxes.max() + margin, W)   
    return [mind, maxd, minh, maxh, minw, maxw]

def get_largest_two_component(img, prt = False, threshold = None):
    s = ndimage.generate_binary_structure(3,2) # iterate structure
    labeled_array, numpatches = ndimage.label(img,s) # labeling
    sizes = ndimage.sum(img,labeled_array,range(1,numpatches+1)) 
    sizes_list = [sizes[i] for i in range(len(sizes))]
    sizes_list.sort()
    if(prt):
        print('component size', sizes_list, flush = True)
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
                print(max_size2, max_size1, max_size2/max_size1, flush = True)   
            if(max_size2*10 > max_size1):
                component1 = (component1 + component2) > 0
            
            return component1

def fill_holes(img): 
    neg = 1 - img
    s = ndimage.generate_binary_structure(3,1) # iterate structure
    labeled_array, numpatches = ndimage.label(neg,s) # labeling
    sizes = ndimage.sum(neg,labeled_array,range(1,numpatches+1)) 
    sizes_list = [sizes[i] for i in range(len(sizes))]
    sizes_list.sort()
    max_size = sizes_list[-1]
    max_label = np.where(sizes == max_size)[0] + 1
    component = labeled_array == max_label
    return 1 - component

def remove_external_core(lab_main, lab_ext):
    # for each component of lab_ext, compute the overlap with lab_main
    s = ndimage.generate_binary_structure(3,2) # iterate structure
    labeled_array, numpatches = ndimage.label(lab_ext,s) # labeling
    sizes = ndimage.sum(lab_ext,labeled_array,range(1,numpatches+1)) 
    sizes_list = [sizes[i] for i in range(len(sizes))]
    new_lab_ext = np.zeros_like(lab_ext)
    for i in range(len(sizes)):
        sizei = sizes_list[i]
        labeli =  np.where(sizes == sizei)[0] + 1
        componenti = labeled_array == labeli
        overlap = componenti * lab_main
        if((overlap.sum()+ 0.0)/sizei >= 0.5):
            new_lab_ext = np.maximum(new_lab_ext, componenti)
    return new_lab_ext

def run(config_file):
    # construct graph
    config = parse_config(config_file)
    config_data = config['data']
    config_net1 = config.get('network1', None)
    config_net2 = config.get('network2', None)
    config_net3 = config.get('network3', None)
    config_test = config['testing']  
    batch_size   = config_test.get('batch_size', 5) 
    if(config_net1):
        net_type1    = config_net1['net_type']
        net_name1    = config_net1['net_name']
        data_shape1  = config_net1['data_shape']
        label_shape1 = config_net1['label_shape']
        data_channel1= config_net1['data_channel']
        class_num1   = config_net1['class_num']
        # construct graph for 1st network
        full_data_shape1 = [batch_size] + data_shape1 + [data_channel1]
        x1 = tf.placeholder(tf.float32, shape = full_data_shape1)          
        net_class1 = NetFactory.create(net_type1)
        net1 = net_class1(num_classes = class_num1,w_regularizer = None,
                    b_regularizer = None, name = net_name1)
        net1.set_params(config_net1)
        predicty1 = net1(x1, is_training = True)
        proby1 = tf.nn.softmax(predicty1)
    else:
        config_net1ax = config['network1ax']
        config_net1sg = config['network1sg']
        config_net1cr = config['network1cr']
        
        # construct graph for 1st network axial
        net_type1ax    = config_net1ax['net_type']
        net_name1ax    = config_net1ax['net_name']
        data_shape1ax  = config_net1ax['data_shape']
        label_shape1ax = config_net1ax['label_shape']
        data_channel1ax= config_net1ax['data_channel']
        class_num1ax   = config_net1ax['class_num']
        
        full_data_shape1ax = [batch_size] + data_shape1ax + [data_channel1ax]
        x1ax = tf.placeholder(tf.float32, shape = full_data_shape1ax)          
        net_class1ax = NetFactory.create(net_type1ax)
        net1ax = net_class1ax(num_classes = class_num1ax,w_regularizer = None,
                    b_regularizer = None, name = net_name1ax)
        net1ax.set_params(config_net1ax)
        predicty1ax = net1ax(x1ax, is_training = True)
        proby1ax = tf.nn.softmax(predicty1ax)

        # construct graph for 1st network sagittal
        net_type1sg    = config_net1sg['net_type']
        net_name1sg    = config_net1sg['net_name']
        data_shape1sg  = config_net1sg['data_shape']
        label_shape1sg = config_net1sg['label_shape']
        data_channel1sg= config_net1sg['data_channel']
        class_num1sg   = config_net1sg['class_num']
        # construct graph for 1st network
        full_data_shape1sg = [batch_size] + data_shape1sg + [data_channel1sg]
        x1sg = tf.placeholder(tf.float32, shape = full_data_shape1sg)          
        net_class1sg = NetFactory.create(net_type1sg)
        net1sg = net_class1sg(num_classes = class_num1sg,w_regularizer = None,
                    b_regularizer = None, name = net_name1sg)
        net1sg.set_params(config_net1sg)
        predicty1sg = net1sg(x1sg, is_training = True)
        proby1sg = tf.nn.softmax(predicty1sg)
            
        # construct graph for 1st network corogal
        net_type1cr    = config_net1cr['net_type']
        net_name1cr    = config_net1cr['net_name']
        data_shape1cr  = config_net1cr['data_shape']
        label_shape1cr = config_net1cr['label_shape']
        data_channel1cr= config_net1cr['data_channel']
        class_num1cr   = config_net1cr['class_num']
        # construct graph for 1st network
        full_data_shape1cr = [batch_size] + data_shape1cr + [data_channel1cr]
        x1cr = tf.placeholder(tf.float32, shape = full_data_shape1cr)          
        net_class1cr = NetFactory.create(net_type1cr)
        net1cr = net_class1cr(num_classes = class_num1cr,w_regularizer = None,
                    b_regularizer = None, name = net_name1cr)
        net1cr.set_params(config_net1cr)
        predicty1cr = net1cr(x1cr, is_training = True)
        proby1cr = tf.nn.softmax(predicty1cr)
    
    # networks for tumor core
    if(config_net2):
        net_type2    = config_net2['net_type']
        net_name2    = config_net2['net_name']
        data_shape2  = config_net2['data_shape']
        label_shape2 = config_net2['label_shape']
        data_channel2= config_net2['data_channel']
        class_num2   = config_net2['class_num']
        # construct graph for 2st network
        full_data_shape2 = [batch_size] + data_shape2 + [data_channel2]
        x2 = tf.placeholder(tf.float32, shape = full_data_shape2)          
        net_class2 = NetFactory.create(net_type2)
        net2 = net_class2(num_classes = class_num2,w_regularizer = None,
                    b_regularizer = None, name = net_name2)
        net2.set_params(config_net2)
        predicty2 = net2(x2, is_training = True)
        proby2 = tf.nn.softmax(predicty2)
    else:
        config_net2ax = config['network2ax']
        config_net2sg = config['network2sg']
        config_net2cr = config['network2cr']
        
        # construct graph for 2st network axial
        net_type2ax    = config_net2ax['net_type']
        net_name2ax    = config_net2ax['net_name']
        data_shape2ax  = config_net2ax['data_shape']
        label_shape2ax = config_net2ax['label_shape']
        data_channel2ax= config_net2ax['data_channel']
        class_num2ax   = config_net2ax['class_num']
        
        full_data_shape2ax = [batch_size] + data_shape2ax + [data_channel2ax]
        x2ax = tf.placeholder(tf.float32, shape = full_data_shape2ax)          
        net_class2ax = NetFactory.create(net_type2ax)
        net2ax = net_class2ax(num_classes = class_num2ax,w_regularizer = None,
                    b_regularizer = None, name = net_name2ax)
        net2ax.set_params(config_net2ax)
        predicty2ax = net2ax(x2ax, is_training = True)
        proby2ax = tf.nn.softmax(predicty2ax)

        # construct graph for 2st network sagittal
        net_type2sg    = config_net2sg['net_type']
        net_name2sg    = config_net2sg['net_name']
        data_shape2sg  = config_net2sg['data_shape']
        label_shape2sg = config_net2sg['label_shape']
        data_channel2sg= config_net2sg['data_channel']
        class_num2sg   = config_net2sg['class_num']
        # construct graph for 2st network
        full_data_shape2sg = [batch_size] + data_shape2sg + [data_channel2sg]
        x2sg = tf.placeholder(tf.float32, shape = full_data_shape2sg)          
        net_class2sg = NetFactory.create(net_type2sg)
        net2sg = net_class2sg(num_classes = class_num2sg,w_regularizer = None,
                    b_regularizer = None, name = net_name2sg)
        net2sg.set_params(config_net2sg)
        predicty2sg = net2sg(x2sg, is_training = True)
        proby2sg = tf.nn.softmax(predicty2sg)
            
        # construct graph for 2st network corogal
        net_type2cr    = config_net2cr['net_type']
        net_name2cr    = config_net2cr['net_name']
        data_shape2cr  = config_net2cr['data_shape']
        label_shape2cr = config_net2cr['label_shape']
        data_channel2cr= config_net2cr['data_channel']
        class_num2cr   = config_net2cr['class_num']
        # construct graph for 2st network
        full_data_shape2cr = [batch_size] + data_shape2cr + [data_channel2cr]
        x2cr = tf.placeholder(tf.float32, shape = full_data_shape2cr)          
        net_class2cr = NetFactory.create(net_type2cr)
        net2cr = net_class2cr(num_classes = class_num2cr,w_regularizer = None,
                    b_regularizer = None, name = net_name2cr)
        net2cr.set_params(config_net2cr)
        predicty2cr = net2cr(x2cr, is_training = True)
        proby2cr = tf.nn.softmax(predicty2cr)

    # for enhanced tumor
    if(config_net3):
        net_type3    = config_net3['net_type']
        net_name3    = config_net3['net_name']
        data_shape3  = config_net3['data_shape']
        label_shape3 = config_net3['label_shape']
        data_channel3= config_net3['data_channel']
        class_num3   = config_net3['class_num']
        # construct graph for 3st network
        full_data_shape3 = [batch_size] + data_shape3 + [data_channel3]
        x3 = tf.placeholder(tf.float32, shape = full_data_shape3)          
        net_class3 = NetFactory.create(net_type3)
        net3 = net_class3(num_classes = class_num3,w_regularizer = None,
                    b_regularizer = None, name = net_name3)
        net3.set_params(config_net3)
        predicty3 = net3(x3, is_training = True)
        proby3 = tf.nn.softmax(predicty3)
    else:
        config_net3ax = config['network3ax']
        config_net3sg = config['network3sg']
        config_net3cr = config['network3cr']
        
        # construct graph for 3st network axial
        net_type3ax    = config_net3ax['net_type']
        net_name3ax    = config_net3ax['net_name']
        data_shape3ax  = config_net3ax['data_shape']
        label_shape3ax = config_net3ax['label_shape']
        data_channel3ax= config_net3ax['data_channel']
        class_num3ax   = config_net3ax['class_num']
        
        full_data_shape3ax = [batch_size] + data_shape3ax + [data_channel3ax]
        x3ax = tf.placeholder(tf.float32, shape = full_data_shape3ax)          
        net_class3ax = NetFactory.create(net_type3ax)
        net3ax = net_class3ax(num_classes = class_num3ax,w_regularizer = None,
                    b_regularizer = None, name = net_name3ax)
        net3ax.set_params(config_net3ax)
        predicty3ax = net3ax(x3ax, is_training = True)
        proby3ax = tf.nn.softmax(predicty3ax)

        # construct graph for 3st network sagittal
        net_type3sg    = config_net3sg['net_type']
        net_name3sg    = config_net3sg['net_name']
        data_shape3sg  = config_net3sg['data_shape']
        label_shape3sg = config_net3sg['label_shape']
        data_channel3sg= config_net3sg['data_channel']
        class_num3sg   = config_net3sg['class_num']
        # construct graph for 3st network
        full_data_shape3sg = [batch_size] + data_shape3sg + [data_channel3sg]
        x3sg = tf.placeholder(tf.float32, shape = full_data_shape3sg)          
        net_class3sg = NetFactory.create(net_type3sg)
        net3sg = net_class3sg(num_classes = class_num3sg,w_regularizer = None,
                    b_regularizer = None, name = net_name3sg)
        net3sg.set_params(config_net3sg)
        predicty3sg = net3sg(x3sg, is_training = True)
        proby3sg = tf.nn.softmax(predicty3sg)
            
        # construct graph for 3st network corogal
        net_type3cr    = config_net3cr['net_type']
        net_name3cr    = config_net3cr['net_name']
        data_shape3cr  = config_net3cr['data_shape']
        label_shape3cr = config_net3cr['label_shape']
        data_channel3cr= config_net3cr['data_channel']
        class_num3cr   = config_net3cr['class_num']
        # construct graph for 3st network
        full_data_shape3cr = [batch_size] + data_shape3cr + [data_channel3cr]
        x3cr = tf.placeholder(tf.float32, shape = full_data_shape3cr)          
        net_class3cr = NetFactory.create(net_type3cr)
        net3cr = net_class3cr(num_classes = class_num3cr,w_regularizer = None,
                    b_regularizer = None, name = net_name3cr)
        net3cr.set_params(config_net3cr)
        predicty3cr = net3cr(x3cr, is_training = True)
        proby3cr = tf.nn.softmax(predicty3cr)
        
    all_vars = tf.global_variables()
    print('all vars', len(all_vars))
    sess = tf.InteractiveSession()   
    sess.run(tf.global_variables_initializer())  
    if(config_net1):
        net1_vars = [x for x in all_vars if x.name[0:len(net_name1) + 1]==net_name1 + '/']
        saver1 = tf.train.Saver(net1_vars)
        saver1.restore(sess, config_net1['model_file'])
    else:
#        net1ax_vars = [x for x in all_vars if x.name[0:len(net_name1ax)+1]==net_name1ax + '/']
#        saver1ax = tf.train.Saver(net1ax_vars)
#
#        saver1ax.restore(sess, config_net1sg['model_file'])
#        net1sg_vars = [x for x in all_vars if x.name[0:len(net_name1sg)+1]==net_name1sg + '/']
#        for i in range(len(net1sg_vars)):
#            copy_value = tf.assign(net1sg_vars[i], net1ax_vars[i])
#            copy_value.eval()
#        print('net1sg loaded')
#        saver1sg = tf.train.Saver(net1sg_vars)
#        saver1sg.save(sess, "model/msnet_wt32sg_20000cp.ckpt")
#        print('netsg saved')
#
#        saver1ax.restore(sess, config_net1cr['model_file'])
#        net1cr_vars = [x for x in all_vars if x.name[0:len(net_name1cr)+1]==net_name1cr + '/']
#        for i in range(len(net1cr_vars)):
#            copy_value = tf.assign(net1cr_vars[i], net1ax_vars[i])
#            copy_value.eval()
#        saver1cr = tf.train.Saver(net1cr_vars)
#        saver1cr.save(sess, "model/msnet_wt32cr_20000cp.ckpt")
#        print('net1cr saved')
#
#        saver1ax.restore(sess, config_net1ax['model_file'])

        net1ax_vars = [x for x in all_vars if x.name[0:len(net_name1ax) + 1]==net_name1ax + '/']
        saver1ax = tf.train.Saver(net1ax_vars)
        print('net1ax', len(net1ax_vars))
        saver1ax.restore(sess, config_net1ax['model_file'])
        net1sg_vars = [x for x in all_vars if x.name[0:len(net_name1sg) + 1]==net_name1sg + '/']
        print('net1sg', len(net1sg_vars))
        saver1sg = tf.train.Saver(net1sg_vars)
        saver1sg.restore(sess, config_net1sg['model_file'])     
        net1cr_vars = [x for x in all_vars if x.name[0:len(net_name1cr) + 1]==net_name1cr + '/']
        saver1cr = tf.train.Saver(net1cr_vars)
        print('net1cr', len(net1cr_vars))
        saver1cr.restore(sess, config_net1cr['model_file'])

    if(config_net2):
        net2_vars = [x for x in all_vars if x.name[0:len(net_name2) + 1]==net_name2 + '/']
        saver2 = tf.train.Saver(net2_vars)
        saver2.restore(sess, config_net2['model_file'])
    else:
        net2ax_vars = [x for x in all_vars if x.name[0:len(net_name2ax)+1]==net_name2ax + '/']
        saver2ax = tf.train.Saver(net2ax_vars)
        saver2ax.restore(sess, config_net2sg['model_file'])
        net2sg_vars = [x for x in all_vars if x.name[0:len(net_name2sg)+1]==net_name2sg + '/']
        for i in range(len(net2sg_vars)):
            copy_value = tf.assign(net2sg_vars[i], net2ax_vars[i])
            copy_value.eval()
        print('net2sg loaded')
        saver2sg = tf.train.Saver(net2sg_vars)
        saver2sg.save(sess, "model/msnet_tc32sg_20000cp.ckpt")
        print('net2sg saved')

        saver2ax.restore(sess, config_net2cr['model_file'])
        net2cr_vars = [x for x in all_vars if x.name[0:len(net_name2cr)+1]==net_name2cr + '/']
        for i in range(len(net2cr_vars)):
            copy_value = tf.assign(net2cr_vars[i], net2ax_vars[i])
            copy_value.eval()
        saver2cr = tf.train.Saver(net2cr_vars)
        saver2cr.save(sess, "model/msnet_tc32cr_20000cp.ckpt")
        print('net2cr saved')

        saver2ax.restore(sess, config_net2ax['model_file'])

#        net2ax_vars = [x for x in all_vars if x.name[0:len(net_name2ax)+1]==net_name2ax + '/']
#        saver2ax = tf.train.Saver(net2ax_vars)
#        saver2ax.restore(sess, config_net2ax['model_file'])
#        net2sg_vars = [x for x in all_vars if x.name[0:len(net_name2sg)+1]==net_name2sg + '/']
#        saver2sg = tf.train.Saver(net2sg_vars)
#        saver2sg.restore(sess, config_net2sg['model_file'])     
#        net2cr_vars = [x for x in all_vars if x.name[0:len(net_name2cr)+1]==net_name2cr + '/']
#        saver2cr = tf.train.Saver(net2cr_vars)
#        saver2cr.restore(sess, config_net2cr['model_file'])     

    if(config_net3):
        net3_vars = [x for x in all_vars if x.name[0:len(net_name3) + 1]==net_name3 + '/']
        saver3 = tf.train.Saver(net3_vars)
        saver3.restore(sess, config_net3['model_file'])
    else:
        net3ax_vars = [x for x in all_vars if x.name[0:len(net_name3ax)+1]==net_name3ax + '/']
        saver3ax = tf.train.Saver(net3ax_vars)
        saver3ax.restore(sess, config_net3sg['model_file'])
        net3sg_vars = [x for x in all_vars if x.name[0:len(net_name3sg)+1]==net_name3sg + '/']
        for i in range(len(net3sg_vars)):
            copy_value = tf.assign(net3sg_vars[i], net3ax_vars[i])
            copy_value.eval()
        print('net3sg loaded')
        saver3sg = tf.train.Saver(net3sg_vars)
        saver3sg.save(sess, "model/msnet_tc32en_20000cp.ckpt")
        print('net3sg saved')

        saver3ax.restore(sess, config_net3cr['model_file'])
        net3cr_vars = [x for x in all_vars if x.name[0:len(net_name3cr)+1]==net_name3cr + '/']
        for i in range(len(net3cr_vars)):
            copy_value = tf.assign(net3cr_vars[i], net3ax_vars[i])
            copy_value.eval()
        saver3cr = tf.train.Saver(net3cr_vars)
        saver3cr.save(sess, "model/msnet_tc32en_20000cp.ckpt")
        print('net3cr saved')
        
        saver3ax.restore(sess, config_net3ax['model_file'])
#        net3ax_vars = [x for x in all_vars if x.name[0:len(net_name3ax) + 1]==net_name3ax+ '/']
#        saver3ax = tf.train.Saver(net3ax_vars)
#        saver3ax.restore(sess, config_net3ax['model_file'])
#        net3sg_vars = [x for x in all_vars if x.name[0:len(net_name3sg) + 1]==net_name3sg+ '/']
#        saver3sg = tf.train.Saver(net3sg_vars)
#        saver3sg.restore(sess, config_net3sg['model_file'])     
#        net3cr_vars = [x for x in all_vars if x.name[0:len(net_name3cr) + 1]==net_name3cr+ '/']
#        saver3cr = tf.train.Saver(net3cr_vars)
#        saver3cr.restore(sess, config_net3cr['model_file'])     

    loader = DataLoader()
    loader.set_params(config_data)
    loader.load_data()
  
    # start to test  
    image_num = loader.get_total_image_number()
    test_slice_direction = config_test.get('test_slice_direction', 'all')
    save_folder = config_test['save_folder']
    test_time = []
    struct = ndimage.generate_binary_structure(3, 2)
    margin = config_test.get('roi_patch_margin', 5)
    for i in range(image_num):
        # test of 1st network
        t0 = time.time()
        [imgs, weight, temp_name] = loader.get_image_data_with_name(i)
        groi = get_roi(weight > 0, margin)
        temp_imgs = [x[np.ix_(range(groi[0], groi[1]), range(groi[2], groi[3]), range(groi[4], groi[5]))] \
                      for x in imgs]
        temp_weight = weight[np.ix_(range(groi[0], groi[1]), range(groi[2], groi[3]), range(groi[4], groi[5]))]

        if(config_net1):
            data_shapes = [data_shape1, data_shape1, data_shape1]
            label_shapes = [label_shape1, label_shape1, label_shape1]
            nets = [net1, net1, net1]
            outputs = [proby1, proby1, proby1]
            inputs =  [x1, x1, x1]
            data_channel = data_channel1
            class_num = class_num1
        else:
            data_shapes = [data_shape1ax, data_shape1sg, data_shape1cr]
            label_shapes = [label_shape1ax, label_shape1sg, label_shape1cr]
            nets = [net1ax, net1sg, net1cr]
            outputs = [proby1ax, proby1sg, proby1cr]
            inputs =  [x1ax, x1sg, x1cr]
            data_channel = data_channel1ax
            class_num = class_num1ax
        prob1 = test_one_image_three_nets_adaptive_shape(temp_imgs, data_shapes, label_shapes, data_channel, class_num,
                   batch_size, sess, nets, outputs, inputs, shape_mode = 0)
        pred1 =  np.asarray(np.argmax(prob1, axis = 3), np.uint16)
        pred1 = pred1 * temp_weight
          
        # test of 2nd network
        wt_threshold = 2000
        if(pred1.sum() == 0):
            print('net1 output is null', temp_name)
            roi2 = get_roi(temp_imgs[0] > 0, margin)
        else:
            pred1_lc = ndimage.morphology.binary_closing(pred1, structure = struct)
            pred1_lc = get_largest_two_component(pred1_lc, True, wt_threshold)
            roi2 = get_roi(pred1_lc, margin)
        sub_imgs = [x[np.ix_(range(roi2[0], roi2[1]), range(roi2[2], roi2[3]), range(roi2[4], roi2[5]))] \
                      for x in temp_imgs]
        sub_weight = temp_weight[np.ix_(range(roi2[0], roi2[1]), range(roi2[2], roi2[3]), range(roi2[4], roi2[5]))]
        if(config_net2):
            data_shapes = [data_shape2, data_shape2, data_shape2]
            label_shapes = [label_shape2, label_shape2, label_shape2]
            nets = [net2, net2, net2]
            outputs = [proby2, proby2, proby2]
            inputs =  [x2, x2, x2]
            data_channel = data_channel2
            class_num = class_num2
        else:
            data_shapes = [data_shape2ax, data_shape2sg, data_shape2cr]
            label_shapes = [label_shape2ax, label_shape2sg, label_shape2cr]
            nets = [net2ax, net2sg, net2cr]
            outputs = [proby2ax, proby2sg, proby2cr]
            inputs =  [x2ax, x2sg, x2cr]
            data_channel = data_channel2ax
            class_num = class_num2ax
        prob2 = test_one_image_three_nets_adaptive_shape(sub_imgs, data_shapes, label_shapes, data_channel, class_num,
                   batch_size, sess, nets, outputs, inputs, shape_mode = 1)
        pred2 = np.asarray(np.argmax(prob2, axis = 3), np.uint16)
        pred2 = pred2 * sub_weight
         
        # test of 3rd network
        if(pred2.sum() == 0):
            [roid, roih, roiw] = sub_imgs[0].shape
            roi3 = [0, roid, 0, roih, 0, roiw]
            subsub_imgs = sub_imgs
            subsub_weight = sub_weight
        else:
            pred2_lc = ndimage.morphology.binary_closing(pred2, structure = struct)
            pred2_lc = get_largest_two_component(pred2_lc)
            roi3 = get_roi(pred2_lc, margin)
            subsub_imgs = [x[np.ix_(range(roi3[0], roi3[1]), range(roi3[2], roi3[3]), range(roi3[4], roi3[5]))] \
                      for x in sub_imgs]
            subsub_weight = sub_weight[np.ix_(range(roi3[0], roi3[1]), range(roi3[2], roi3[3]), range(roi3[4], roi3[5]))] 
        
        if(config_net3):
            data_shapes = [data_shape3, data_shape3, data_shape3]
            label_shapes = [label_shape3, label_shape3, label_shape3]
            nets = [net3, net3, net3]
            outputs = [proby3, proby3, proby3]
            inputs =  [x3, x3, x3]
            data_channel = data_channel3
            class_num = class_num3
        else:
            data_shapes = [data_shape3ax, data_shape3sg, data_shape3cr]
            label_shapes = [label_shape3ax, label_shape3sg, label_shape3cr]
            nets = [net3ax, net3sg, net3cr]
            outputs = [proby3ax, proby3sg, proby3cr]
            inputs =  [x3ax, x3sg, x3cr]
            data_channel = data_channel3ax
            class_num = class_num3ax
        prob3 = test_one_image_three_nets_adaptive_shape(subsub_imgs, data_shapes, label_shapes, data_channel, class_num,
                   batch_size, sess, nets, outputs, inputs, shape_mode = 1)
        
        pred3 = np.asarray(np.argmax(prob3, axis = 3), np.uint16)
        pred3 = pred3 * subsub_weight
         
        # fuse results at 3 levels
        # convert subsub_label to full size (non-enhanced)
        label3_roi = np.zeros_like(pred2)
        label3_roi[np.ix_(range(roi3[0], roi3[1]), range(roi3[2], roi3[3]), range(roi3[4], roi3[5]))] = pred3
        label3 = np.zeros_like(pred1)
        label3[np.ix_(range(roi2[0], roi2[1]), range(roi2[2], roi2[3]), range(roi2[4], roi2[5]))] = label3_roi

        # convert sub_label to full size (tumor core)
        label2 = np.zeros_like(pred1)
        label2[np.ix_(range(roi2[0], roi2[1]), range(roi2[2], roi2[3]), range(roi2[4], roi2[5]))] = pred2

        
        # fuse the results
        label1_mask = (pred1 + label2 + label3) > 0
        label1_mask = ndimage.morphology.binary_closing(label1_mask, structure = struct)
        label1_mask = get_largest_two_component(label1_mask, False, wt_threshold)
        label1 = pred1 * label1_mask
        
        label2_3_mask = (label2 + label3) > 0
        label2_3_mask = label2_3_mask * label1_mask
        label2_3_mask = ndimage.morphology.binary_closing(label2_3_mask, structure = struct)
        label2_3_mask = remove_external_core(label1, label2_3_mask)
        if(label2_3_mask.sum() > 0):
            label2_3_mask = get_largest_two_component(label2_3_mask)
        label1 = (label1 + label2_3_mask) > 0
        label2 = label2_3_mask
        label3 = label2 * label3
        vox_3  = label3.sum() 
        if(0 < vox_3 and vox_3 < 30):
            print('ignored voxel number ', vox_3, flush = True)
            label3 = np.zeros_like(label2)
            
        out_label = label1 * 2 
        out_label[label2>0] = 1
        out_label[label3>0] = 4
        out_label = np.asarray(out_label, np.int16)

        test_time.append(time.time() - t0)
        final_label = np.zeros_like(weight, np.int16)
        final_label[np.ix_(range(groi[0], groi[1]), range(groi[2], groi[3]), range(groi[4], groi[5]))] = out_label
        save_array_as_nifty_volume(final_label, save_folder+"/{0:}.nii.gz".format(temp_name))
        print(temp_name, flush = True)
    test_time = np.asarray(test_time)
    print('test time', test_time.mean(), flush= True)
    np.savetxt(save_folder + '/test_time.txt', test_time)
    sess.close()
      
if __name__ == '__main__':
    if(len(sys.argv) != 2):
        print('Number of arguments should be 2. e.g.')
        print('    python run_test.py config.txt')
        exit()
    config_file = str(sys.argv[1])
    assert(os.path.isfile(config_file))
    run(config_file)
    
    
