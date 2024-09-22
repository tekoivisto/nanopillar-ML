# Converting tensorflow built-in 2D resnet to 3D

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, GlobalAveragePooling3D, ZeroPadding3D, BatchNormalization, Add, Activation
import numpy as np

def get_model(input_shape, dropout=0.0, filter_divider=1, smaller_stride=False):

    # getting past tensorflows built in minimum size for input to resnet50
    if smaller_stride:
        input_shape = np.array(input_shape, dtype=int)
        input_shape[:3] *= 2
        input_shape = tuple(input_shape)

    shape_2d = input_shape[1:]

    premaid_2d = tf.keras.applications.ResNet50(include_top=False, pooling="avg", weights=None, input_shape=shape_2d)

    premaid_3d = convert_2d_resnet_to_3d(premaid_2d, input_shape, filter_divider, smaller_stride)
    model_3d = tf.keras.models.Sequential()
    model_3d.add(premaid_3d)
    model_3d.add(tf.keras.layers.Dropout(dropout))
    model_3d.add(tf.keras.layers.Dense(1))
    return model_3d

def convert_2d_resnet_to_3d(_2d_resnet, input_shape, filter_divider=1, smaller_stride=False):

    if smaller_stride:
        input_shape = np.array(input_shape, dtype=int)
        input_shape[:3] = input_shape[:3]/2
        input_shape = tuple(input_shape)

    all_layers = {}
    input = Input(input_shape, dtype='float32')
    all_layers['input_1'] = input

    # Iterate through each layer in the original 2D model
    for layer in _2d_resnet.layers:
        
        config_dict = layer.get_config()
        
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue

        elif isinstance(layer, tf.keras.layers.Conv2D):
            config_dict = extend_config_by_1(config_dict, 'kernel_size')
            config_dict = extend_config_by_1(config_dict, 'strides')
            config_dict = extend_config_by_1(config_dict, 'dilation_rate')

            config_dict['filters'] = int(config_dict['filters']/filter_divider)
            if layer.name == 'conv1_conv' and smaller_stride:
                config_dict['strides'] = (1, 1, 1)
            new_layer = Conv3D(**config_dict)

            

        elif isinstance(layer, tf.keras.layers.ZeroPadding2D):
            config_dict = extend_config_by_1(config_dict, 'padding')
            new_layer = ZeroPadding3D(**config_dict)
        
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            config_dict['axis'] = -1
            new_layer = BatchNormalization(**config_dict)

        elif isinstance(layer, tf.keras.layers.MaxPooling2D):
            config_dict = extend_config_by_1(config_dict, 'pool_size')
            config_dict = extend_config_by_1(config_dict, 'strides')

            new_layer = MaxPooling3D(**config_dict)

        elif isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):

            new_layer = GlobalAveragePooling3D(**config_dict)
        
        elif isinstance(layer, tf.keras.layers.Add):
            new_inputs = []
            old_inputs = layer.input

            for i in old_inputs:
                #print(i)
                #print(i.name)
                name = i.name
                name = name.split('/')[0]

                new_inputs.append(all_layers[name])

            x = Add(**config_dict)(new_inputs)
            all_layers[layer.name] = x
        
        elif isinstance(layer, tf.keras.layers.Activation):
            new_layer = Activation(**config_dict)

        
        if not isinstance(layer, tf.keras.layers.Add):
            input_name = layer.input.name.split('/')[0]
            #print(input_name)

            x = new_layer(all_layers[input_name])

            all_layers[new_layer.name] = x
    
    model_3d = Model(inputs=input, outputs=x)

    return model_3d

def extend_config_by_1(config_dict, key):
    x = list(config_dict[key])
    x.append(x[0])
    x = tuple(x)
    config_dict[key] = x

    return config_dict

