# Computing Grad-CAM fields. Code adapted from https://github.com/ComplexityBiosystems/2D-silica-ML

import tensorflow as tf
import numpy as np


def get_gc_heatmaps_parallel(model, input_pillars, conv_block, hi_res=False):
    if conv_block==1:
        last_conv_layer = model.get_layer('model').get_layer('conv2_block3_out')
        last_conv_layer_model = tf.keras.Model(model.get_layer('model').inputs, last_conv_layer.output)
        
        classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])

        x = classifier_input

        x = build_conv_block(model, x, block_number=2)
        x = build_conv_block(model, x, block_number=3)
        x = build_conv_block(model, x, block_number=4)

        classifier_model = tf.keras.Model(classifier_input, x)
    
    elif conv_block == 2:
        last_conv_layer = model.get_layer('model').get_layer('conv3_block4_out')
        last_conv_layer_model = tf.keras.Model(model.get_layer('model').inputs, last_conv_layer.output)
        
        classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])

        x = classifier_input

        x = build_conv_block(model, x, block_number=3)
        x = build_conv_block(model, x, block_number=4)

        classifier_model = tf.keras.Model(classifier_input, x)

    elif conv_block == 3:
        last_conv_layer = model.get_layer('model').get_layer('conv4_block6_out')
        last_conv_layer_model = tf.keras.Model(model.get_layer('model').inputs, last_conv_layer.output)
        
        classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])

        x = classifier_input

        x = build_conv_block(model, x, block_number=4)

        classifier_model = tf.keras.Model(classifier_input, x)
    
    elif conv_block == 4:
        last_conv_layer = model.get_layer('model').get_layer('conv5_block3_out')
        last_conv_layer_model = tf.keras.Model(model.get_layer('model').inputs, last_conv_layer.output)
        
        classifier_input = tf.keras.Input(shape = last_conv_layer.output.shape[1:])
        x = classifier_input
        x = model.get_layer('model').get_layer('avg_pool')(x) 
        x = model.get_layer('dropout')(x)
        x = model.get_layer('dense')(x)
        classifier_model = tf.keras.Model(classifier_input, x)

    with tf.GradientTape() as tape:
    
        last_conv_layer_output = last_conv_layer_model(input_pillars)
        tape.watch(last_conv_layer_output)
   
        preds = classifier_model(last_conv_layer_output)
    
    grads = tape.gradient(preds, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis = (1, 2, 3))

    pooled_grads = pooled_grads.numpy()
    grads = grads.numpy()
    last_conv_layer_output = last_conv_layer_output.numpy()

    if hi_res:
        # HiResCam https://arxiv.org/abs/2011.08891 was tried in addition to Grad-CAM, but Grad-CAM produced
        # more sensible results for our dataset
        last_conv_layer_output *= grads
        heatmaps = np.sum(last_conv_layer_output, axis = -1)

    else:
        for i_pillar in range(pooled_grads.shape[0]):
            for i_filter in range(pooled_grads.shape[-1]):
                last_conv_layer_output[i_pillar, :, :, :, i_filter] *= pooled_grads[i_pillar, i_filter]
        
        # This should be sum instead of mean. Doesn't really matter tho
        heatmaps = np.mean(last_conv_layer_output, axis = -1)

    # removed taking absolute
    return heatmaps


def build_conv_block(model, x, block_number):

    if block_number == 2:

        xc1 = model.get_layer('model').get_layer('conv3_block1_1_conv')(x)
        xc2 = model.get_layer('model').get_layer('conv3_block1_1_bn')(xc1) 
        xc3 = model.get_layer('model').get_layer('conv3_block1_1_relu')(xc2)
        xc4 = model.get_layer('model').get_layer('conv3_block1_2_conv')(xc3)
        xc5 = model.get_layer('model').get_layer('conv3_block1_2_bn')(xc4)
        xc6 = model.get_layer('model').get_layer('conv3_block1_2_relu')(xc5)
        xc7 = model.get_layer('model').get_layer('conv3_block1_0_conv')(x)
        xc8 = model.get_layer('model').get_layer('conv3_block1_3_conv')(xc6)
        xc9 = model.get_layer('model').get_layer('conv3_block1_0_bn')(xc7)
        xc10 = model.get_layer('model').get_layer('conv3_block1_3_bn')(xc8)
        xc11 = model.get_layer('model').get_layer('conv3_block1_add')([xc9, xc10])
        xc12 = model.get_layer('model').get_layer('conv3_block1_out')(xc11)
        xc13 = model.get_layer('model').get_layer('conv3_block2_1_conv')(xc12)
        xc14 = model.get_layer('model').get_layer('conv3_block2_1_bn')(xc13)
        xc15 = model.get_layer('model').get_layer('conv3_block2_1_relu')(xc14)
        xc16 = model.get_layer('model').get_layer('conv3_block2_2_conv')(xc15)
        xc17 = model.get_layer('model').get_layer('conv3_block2_2_bn')(xc16)
        xc18 = model.get_layer('model').get_layer('conv3_block2_2_relu')(xc17)
        xc19 = model.get_layer('model').get_layer('conv3_block2_3_conv')(xc18)
        xc20 = model.get_layer('model').get_layer('conv3_block2_3_bn')(xc19)
        xc21 = model.get_layer('model').get_layer('conv3_block2_add')([xc12, xc20])
        xc22 = model.get_layer('model').get_layer('conv3_block2_out')(xc21)
        xc23 = model.get_layer('model').get_layer('conv3_block3_1_conv')(xc22)
        xc24 = model.get_layer('model').get_layer('conv3_block3_1_bn')(xc23)
        xc25 = model.get_layer('model').get_layer('conv3_block3_1_relu')(xc24)
        xc26 = model.get_layer('model').get_layer('conv3_block3_2_conv')(xc25)
        xc27 = model.get_layer('model').get_layer('conv3_block3_2_bn')(xc26)
        xc28 = model.get_layer('model').get_layer('conv3_block3_2_relu')(xc27)
        xc29 = model.get_layer('model').get_layer('conv3_block3_3_conv')(xc28)
        xc30 = model.get_layer('model').get_layer('conv3_block3_3_bn')(xc29)
        xc31 = model.get_layer('model').get_layer('conv3_block3_add')([xc22, xc30])
        xc32 = model.get_layer('model').get_layer('conv3_block3_out')(xc31)
        xc33 = model.get_layer('model').get_layer('conv3_block4_1_conv')(xc32)
        xc34 = model.get_layer('model').get_layer('conv3_block4_1_bn')(xc33)
        xc35 = model.get_layer('model').get_layer('conv3_block4_1_relu')(xc34)
        xc36 = model.get_layer('model').get_layer('conv3_block4_2_conv')(xc35)
        xc37 = model.get_layer('model').get_layer('conv3_block4_2_bn')(xc36)
        xc38 = model.get_layer('model').get_layer('conv3_block4_2_relu')(xc37)
        xc39 = model.get_layer('model').get_layer('conv3_block4_3_conv')(xc38)
        xc40 = model.get_layer('model').get_layer('conv3_block4_3_bn')(xc39)
        xc41 = model.get_layer('model').get_layer('conv3_block4_add')([xc32, xc40])
        xc42 = model.get_layer('model').get_layer('conv3_block4_out')(xc41)

        return xc42

    elif block_number == 3:

        x1 = model.get_layer('model').get_layer('conv4_block1_1_conv')(x)
        x2 = model.get_layer('model').get_layer('conv4_block1_1_bn')(x1)
        x3 = model.get_layer('model').get_layer('conv4_block1_1_relu')(x2)
        x4 = model.get_layer('model').get_layer('conv4_block1_2_conv')(x3)
        x5 = model.get_layer('model').get_layer('conv4_block1_2_bn')(x4)
        x6 = model.get_layer('model').get_layer('conv4_block1_2_relu')(x5)
        x7 = model.get_layer('model').get_layer('conv4_block1_0_conv')(x)
        x8 = model.get_layer('model').get_layer('conv4_block1_3_conv')(x6)
        x9 = model.get_layer('model').get_layer('conv4_block1_0_bn')(x7)
        x10 = model.get_layer('model').get_layer('conv4_block1_3_bn')(x8)
        x11 = model.get_layer('model').get_layer('conv4_block1_add')([x9, x10])
        x12 = model.get_layer('model').get_layer('conv4_block1_out')(x11)
        x13 = model.get_layer('model').get_layer('conv4_block2_1_conv')(x12)
        x14 = model.get_layer('model').get_layer('conv4_block2_1_bn')(x13)
        x15 = model.get_layer('model').get_layer('conv4_block2_1_relu')(x14)
        x16 = model.get_layer('model').get_layer('conv4_block2_2_conv')(x15)
        x17 = model.get_layer('model').get_layer('conv4_block2_2_bn')(x16)
        x18 = model.get_layer('model').get_layer('conv4_block2_2_relu')(x17)
        x19 = model.get_layer('model').get_layer('conv4_block2_3_conv')(x18)
        x20 = model.get_layer('model').get_layer('conv4_block2_3_bn')(x19)
        x21 = model.get_layer('model').get_layer('conv4_block2_add')([x12, x20])
        x22 = model.get_layer('model').get_layer('conv4_block2_out')(x21)
        x23 = model.get_layer('model').get_layer('conv4_block3_1_conv')(x22)
        x24 = model.get_layer('model').get_layer('conv4_block3_1_bn')(x23)
        x25 = model.get_layer('model').get_layer('conv4_block3_1_relu')(x24)
        x26 = model.get_layer('model').get_layer('conv4_block3_2_conv')(x25)
        x27 = model.get_layer('model').get_layer('conv4_block3_2_bn')(x26)
        x28 = model.get_layer('model').get_layer('conv4_block3_2_relu')(x27)
        x29 = model.get_layer('model').get_layer('conv4_block3_3_conv')(x28)
        x30 = model.get_layer('model').get_layer('conv4_block3_3_bn')(x29)
        x31 = model.get_layer('model').get_layer('conv4_block3_add')([x22, x30])
        x32 = model.get_layer('model').get_layer('conv4_block3_out')(x31)
        x33 = model.get_layer('model').get_layer('conv4_block4_1_conv')(x32)
        x34 = model.get_layer('model').get_layer('conv4_block4_1_bn')(x33)
        x35 = model.get_layer('model').get_layer('conv4_block4_1_relu')(x34)
        x36 = model.get_layer('model').get_layer('conv4_block4_2_conv')(x35)
        x37 = model.get_layer('model').get_layer('conv4_block4_2_bn')(x36)
        x38 = model.get_layer('model').get_layer('conv4_block4_2_relu')(x37)
        x39 = model.get_layer('model').get_layer('conv4_block4_3_conv')(x38)
        x40 = model.get_layer('model').get_layer('conv4_block4_3_bn')(x39)
        x41 = model.get_layer('model').get_layer('conv4_block4_add')([x32, x40])
        x42 = model.get_layer('model').get_layer('conv4_block4_out')(x41)
        x43 = model.get_layer('model').get_layer('conv4_block5_1_conv')(x42)
        x44 = model.get_layer('model').get_layer('conv4_block5_1_bn')(x43)
        x45 = model.get_layer('model').get_layer('conv4_block5_1_relu')(x44)
        x46 = model.get_layer('model').get_layer('conv4_block5_2_conv')(x45)
        x47 = model.get_layer('model').get_layer('conv4_block5_2_bn')(x46)
        x48 = model.get_layer('model').get_layer('conv4_block5_2_relu')(x47)
        x49 = model.get_layer('model').get_layer('conv4_block5_3_conv')(x48)
        x50 = model.get_layer('model').get_layer('conv4_block5_3_bn')(x49)
        x51 = model.get_layer('model').get_layer('conv4_block5_add')([x42, x50])
        x52 = model.get_layer('model').get_layer('conv4_block5_out')(x51)
        x53 = model.get_layer('model').get_layer('conv4_block6_1_conv')(x52)
        x54 = model.get_layer('model').get_layer('conv4_block6_1_bn')(x53)
        x55 = model.get_layer('model').get_layer('conv4_block6_1_relu')(x54)
        x56 = model.get_layer('model').get_layer('conv4_block6_2_conv')(x55)
        x57 = model.get_layer('model').get_layer('conv4_block6_2_bn')(x56)
        x58 = model.get_layer('model').get_layer('conv4_block6_2_relu')(x57)
        x59 = model.get_layer('model').get_layer('conv4_block6_3_conv')(x58)
        x60 = model.get_layer('model').get_layer('conv4_block6_3_bn')(x59)
        x61 = model.get_layer('model').get_layer('conv4_block6_add')([x52, x60])
        x62 = model.get_layer('model').get_layer('conv4_block6_out')(x61)

        return x62

    elif block_number == 4:

        x1 = model.get_layer('model').get_layer('conv5_block1_1_conv')(x)
        x2 = model.get_layer('model').get_layer('conv5_block1_1_bn')(x1)
        x3 = model.get_layer('model').get_layer('conv5_block1_1_relu')(x2)
        x4 = model.get_layer('model').get_layer('conv5_block1_2_conv')(x3)
        x5 = model.get_layer('model').get_layer('conv5_block1_2_bn')(x4)
        x6 = model.get_layer('model').get_layer('conv5_block1_2_relu')(x5)
        x7 = model.get_layer('model').get_layer('conv5_block1_0_conv')(x)
        x8 = model.get_layer('model').get_layer('conv5_block1_3_conv')(x6)
        x9 = model.get_layer('model').get_layer('conv5_block1_0_bn')(x7)
        x10 = model.get_layer('model').get_layer('conv5_block1_3_bn')(x8)
        x11 = model.get_layer('model').get_layer('conv5_block1_add')([x9,x10])
        x12 = model.get_layer('model').get_layer('conv5_block1_out')(x11)
        x13 = model.get_layer('model').get_layer('conv5_block2_1_conv')(x12)
        x14 = model.get_layer('model').get_layer('conv5_block2_1_bn')(x13)
        x15 = model.get_layer('model').get_layer('conv5_block2_1_relu')(x14)
        x16 = model.get_layer('model').get_layer('conv5_block2_2_conv')(x15)
        x17 = model.get_layer('model').get_layer('conv5_block2_2_bn')(x16)
        x18 = model.get_layer('model').get_layer('conv5_block2_2_relu')(x17)
        x19 = model.get_layer('model').get_layer('conv5_block2_3_conv')(x18)
        x20 = model.get_layer('model').get_layer('conv5_block2_3_bn')(x19)
        x21 = model.get_layer('model').get_layer('conv5_block2_add')([x12,x20])
        x22 = model.get_layer('model').get_layer('conv5_block2_out')(x21)
        x23 = model.get_layer('model').get_layer('conv5_block3_1_conv')(x22)
        x24 = model.get_layer('model').get_layer('conv5_block3_1_bn')(x23)
        x25 = model.get_layer('model').get_layer('conv5_block3_1_relu')(x24)
        x26 = model.get_layer('model').get_layer('conv5_block3_2_conv')(x25)
        x27 = model.get_layer('model').get_layer('conv5_block3_2_bn')(x26)
        x28 = model.get_layer('model').get_layer('conv5_block3_2_relu')(x27)
        x29 = model.get_layer('model').get_layer('conv5_block3_3_conv')(x28)
        x30 = model.get_layer('model').get_layer('conv5_block3_3_bn')(x29)
        x31 = model.get_layer('model').get_layer('conv5_block3_add')([x21,x30])
        x32 = model.get_layer('model').get_layer('conv5_block3_out')(x31)
        x33 = model.get_layer('model').get_layer('avg_pool')(x32)
        x34 = model.get_layer('dropout')(x33)
        #x35 = model.get_layer('Output')(x34)
        x35 = model.get_layer('dense')(x34)

        return x35
