# Trains and evaluates the CNNs and generates the Grad-CAM fields

import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import layers, callbacks, optimizers, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow_addons.metrics import RSquare
import random as python_random
import argparse
from time import time
import datetime
import pickle
import h5py
from resnet50 import get_model
from data_generator import DataGenerator, augment_n_flip_parallel
from get_gc_heatmap import get_gc_heatmaps_parallel

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-s', '--seed', type=int, help='seed for random number generator')
arg_parser.add_argument('-g', '--ground_truth', type=str, help='filename for ground truth value')
arg_parser.add_argument('-d', '--descriptors', type=str, help='descriptors,raw, grain_boundary, quaternion or combined')
arg_parser.add_argument('-i', '--index', type=int, help='index if ground truth is stresses_at_points')
arg_parser.add_argument('--num_epochs', type=int)
arg_parser.add_argument('--resolution', type=int)
arg_parser.add_argument('--patience', type=int)
arg_parser.add_argument('--learning_rate', type=float)
arg_parser.add_argument('--filter_divider', type=int)
arg_parser.add_argument('--batch_size', type=int, default=32)
arg_parser.add_argument('--grad_cam_batch_size', type=int, default=8)
arg_parser.add_argument('--dropout', type=float, default=0.5)

seed = arg_parser.parse_args().seed
gt_file = arg_parser.parse_args().ground_truth
descriptor_type = arg_parser.parse_args().descriptors
stresses_at_point_idx = arg_parser.parse_args().index
num_epochs = arg_parser.parse_args().num_epochs
res = arg_parser.parse_args().resolution
patience = arg_parser.parse_args().patience
learning_rate = arg_parser.parse_args().learning_rate
filter_divider = arg_parser.parse_args().filter_divider
batch_size = arg_parser.parse_args().batch_size
grad_cam_batch_size = arg_parser.parse_args().grad_cam_batch_size
dropout = arg_parser.parse_args().dropout

def r_2_score(y_true, y_pred):
    RSS =  K.sum(K.square(y_true - y_pred))
    TSS = K.sum(K.square(y_true - K.mean(y_true)))
    return ( 1 - (RSS/TSS) )

def r_2_score_numpy(y_true, y_pred):
    RSS =  np.sum(np.square(y_true - y_pred))
    TSS = np.sum(np.square(y_true - np.mean(y_true)))
    return ( 1 - (RSS/TSS) )

def evaluate_with_data_augmentation(X,y, model):
    
    preds_all = []

    for augmentation in range(16):
        generator = DataGenerator(y, X,
                                  batch_size=batch_size, dim=(size_X, size_Y, size_Z),
                                  n_channels=n_channels, random_augment=False,
                                  constant_augment=augmentation, shuffle=False)
        
        preds = model.predict(generator, batch_size=batch_size)
        preds = preds.reshape(y.size)
        preds_all.append(preds)

    avg_preds = np.mean(preds_all, axis=0)
    r_2_score = r_2_score_numpy(y, avg_preds)

    return avg_preds, r_2_score

def generate_grad_cam_heatmaps(model, pillars, grad_cam_batch_size, seed):
    
    hm_dir_name = f'grad_cam_heatmaps_{seed}'
    os.makedirs(hm_dir_name, exist_ok=True)

    if type(pillars) == tuple:
        N_pillars = pillars[0].shape[0]
        pillar_resolution = pillars[0].shape[1]
    else:
        N_pillars = pillars.shape[0]
        pillar_resolution = pillars.shape[1]


    for conv_block in range(1,5):
        
        if res < 32:
            r = pillar_resolution // 2**conv_block
        else:
            r = pillar_resolution // 2**(conv_block+1)
        shape = (N_pillars, r, r, 2*r)

        all_heatmaps_augmented = np.zeros(shape, dtype=np.float32)

        print('conv_block', conv_block)
        
        for i_pillar in range(0, N_pillars, grad_cam_batch_size):

            end_idx = min(i_pillar + grad_cam_batch_size, N_pillars)
            if type(pillars) == tuple:
                batch_pillars1 = pillars[0][i_pillar:end_idx]
                batch_pillars2 = pillars[1][i_pillar:end_idx]
                batch_pillars = np.concatenate((batch_pillars1, batch_pillars2), axis=-1)
            else:
                batch_pillars = pillars[i_pillar:end_idx]
            

            augmentations = []
            for n in range(16):
                batch_pillars_aug = augment_n_flip_parallel(batch_pillars, n)
                batch_heatmaps_aug = get_gc_heatmaps_parallel(model, batch_pillars_aug, conv_block)

                batch_heatmaps_aug = augment_n_flip_parallel(batch_heatmaps_aug, -n)
                augmentations.append(batch_heatmaps_aug)

            batch_heatmaps_averaged_aug = np.average(augmentations, axis=0)
            all_heatmaps_augmented[i_pillar:end_idx] = batch_heatmaps_averaged_aug

        np.save(f'{hm_dir_name}/conv_block_augmented_{conv_block}.npy', all_heatmaps_augmented)


python_random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

path = '../'
while 'descriptors' not in os.listdir(path):
    path +='../'

size_X = res
size_Y = res
size_Z = 2*res

print('resolution', res)
print(descriptor_type)

if descriptor_type != 'combined':

    if descriptor_type == 'raw':
        n_channels = 1
        dataset = h5py.File(f'{path}descriptors/pillars_brigthness_function_1.04_{res}.h5','r')

    elif descriptor_type == 'grain_boundary':
        n_channels = 1
        dataset = h5py.File(f'{path}descriptors/grain_boundary_{res}.h5','r')

    elif descriptor_type == 'quaternion':
        n_channels = 4
        dataset = h5py.File(f'{path}descriptors/quaternion_{res}.h5','r')
    
    training_set = dataset['training_set']
    validation_set = dataset['validation_set']
    test_set = dataset['test_set']


else:
    n_channels = 5
    dataset_g = h5py.File(f'{path}descriptors/grain_boundary_{res}.h5','r')
    dataset_q = h5py.File(f'{path}descriptors/quaternion_{res}.h5','r')

    training_set = (dataset_g['training_set'], dataset_q['training_set'])
    validation_set = (dataset_g['validation_set'], dataset_q['validation_set'])
    test_set = (dataset_g['test_set'], dataset_q['test_set'])


val_n = 500

Y = np.load(f'{path}ground_truth/tr_val_set/{gt_file}')
Y_test = np.load(f'{path}ground_truth/test_set/{gt_file}')

mean_y = np.mean(Y)
std_y = np.std(Y)

Y = (Y-mean_y)/std_y
Y_test = (Y_test-mean_y)/std_y

Y_val = Y[:val_n]
Y_train = Y[val_n:]

del Y

generator_train=DataGenerator(Y_train, training_set,
                              batch_size=batch_size, dim=(size_X, size_Y, size_Z),
                              n_channels=n_channels, 
                              random_augment=True, constant_augment=0)
generator_val=DataGenerator(Y_val, validation_set,
                              batch_size=batch_size, dim=(size_X, size_Y, size_Z),
                              n_channels=n_channels, 
                              random_augment=False, constant_augment=0)
generator_test=DataGenerator(Y_test, test_set,
                              batch_size=batch_size, dim=(size_X, size_Y, size_Z),
                              n_channels=n_channels, 
                              random_augment=False, constant_augment=0)

print('created generators successfully')

input_shape = (size_X, size_Y, size_Z, n_channels)
print('filter divider', filter_divider)
if res < 32:
    smaller_stride = True
else:
    smaller_stride = False
model = get_model(input_shape, dropout=dropout, filter_divider=filter_divider, smaller_stride=smaller_stride)

model.summary()

opt = optimizers.Adam(learning_rate=learning_rate)
print('learning_rate', learning_rate)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=[RSquare(dtype=tf.float32, y_shape=(1,))])

start_time = time()

es = callbacks.EarlyStopping(monitor = 'val_r_square', mode = 'max', patience = patience, restore_best_weights=True)
fname = 'weights_{epoch:06d}.h5'
best_weights_checkpoint = callbacks.ModelCheckpoint(
                    f'best_weights_{seed}.h5', 
                    monitor='val_r_square', mode = 'max', 
                    save_best_only=True, verbose=1)

print('seed', seed)
print('gt_file', gt_file)

history = model.fit(generator_train, epochs=num_epochs, batch_size=batch_size, validation_data = generator_val, callbacks = [es, best_weights_checkpoint], verbose=2)

model.load_weights(f'best_weights_{seed}.h5')

train_time = time()-start_time

train_score = model.evaluate(generator_train, batch_size=batch_size)[1]
test_score = model.evaluate(generator_test, batch_size=batch_size)[1]
validation_score = model.evaluate(generator_val, batch_size=batch_size)[1]

print("Train score: " + str(train_score))
print("Test score: " + str(test_score))
print("Validation score: " + str(validation_score))

stats = open(f'stats_CNN_{seed}.dat', 'w')
stats.write('#training_set_score validation_set_score test_set_score\n')
stats.write(f'{train_score} {validation_score} {test_score}\n')

_, train_score_augmented = evaluate_with_data_augmentation(training_set, Y_train, model)
_, validation_score_augmented = evaluate_with_data_augmentation(validation_set, Y_val, model)
preds_test_set_augmented, test_score_augmented = evaluate_with_data_augmentation(test_set, Y_test, model)
preds_test_set_augmented = preds_test_set_augmented * std_y + mean_y
np.save(f'preds_test_set_augmented_{seed}.npy', preds_test_set_augmented)

stats.write('#training_set_score_augmented validation_set_score_augmented test_set_score_augmented\n')
stats.write(f'{train_score_augmented} {validation_score_augmented} {test_score_augmented}\n')

stats.write('#n epochs\n')
stats.write(f'{es.stopped_epoch}\n')
stats.write('training time\n')
stats.write(f'{datetime.timedelta(seconds=train_time)}\n')

with open(f'history_{seed}.pkl', 'wb') as f:
    pickle.dump(history.history, f)


stats.close()

generate_grad_cam_heatmaps(model, test_set, grad_cam_batch_size, seed)

dataset.close()
