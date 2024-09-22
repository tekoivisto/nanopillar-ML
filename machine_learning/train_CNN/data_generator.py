# Data generator that reads the descriptors from .h5 files during training and
# evaluation

import tensorflow as tf
import numpy as np

class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, ground_truth, X_h5,  batch_size=32, dim=(128,128,256), n_channels=1,
                 random_augment=False, constant_augment=0, shuffle=True):

        self.dim = dim
        self.batch_size = batch_size
        self.ground_truth = ground_truth

        self.n_channels = n_channels
        self.shuffle = shuffle

        if type(X_h5) == tuple:
            self.X_h5 = X_h5[0]
            self.X_h5_2 = X_h5[1]
            self.multiple_files = True
        else:
            self.X_h5 = X_h5
            self.X_h5_2 = None
            self.multiple_files = False

        self.random_augment = random_augment
        self.constant_augment = constant_augment

        self.indexes = np.arange(len(self.X_h5))
        self.on_epoch_end()

    def __len__(self):

        return int(np.ceil(len(self.X_h5) / self.batch_size))

    def __getitem__(self, index):
        
        start = index*self.batch_size
        end = min((index + 1) * self.batch_size, len(self.X_h5))
        indexes = self.indexes[start:end]
        indexes = np.sort(indexes)

        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        
        if (not self.random_augment) and (not self.constant_augment):
            if not self.multiple_files:
                X = self.X_h5[indexes]
            else:
                X1 = self.X_h5[indexes]
                X2 = self.X_h5_2[indexes]
                X = np.concatenate((X1, X2), axis=-1)

        elif self.random_augment:
            X = np.zeros([len(indexes)]+list(self.dim)+[self.n_channels], dtype=np.float32)
            augmentations = np.random.randint(0, high=16, size=self.batch_size)
            for i_batch, (i_dataset, a) in enumerate(zip(indexes, augmentations)):
                if not self.multiple_files:
                    X[i_batch] = augment_n_flip(self.X_h5[i_dataset], a)
                else:
                    x1 = augment_n_flip(self.X_h5[i_dataset], a)
                    x2 = augment_n_flip(self.X_h5_2[i_dataset], a)
                    X[i_batch] = np.concatenate((x1,x2), axis=-1)

        
        elif self.constant_augment:
            if not self.multiple_files:
                X = augment_n_flip_parallel(self.X_h5[indexes], self.constant_augment)
            else:
                X1 = augment_n_flip_parallel(self.X_h5[indexes], self.constant_augment)
                X2 = augment_n_flip_parallel(self.X_h5_2[indexes], self.constant_augment)
                X = np.concatenate((X1, X2), axis=-1)

        y = self.ground_truth[indexes]

        return X, y

def augment_n_flip(pillar, n):
    # does data augmentation for a single pillar. For positive n, n in range [0, 15].
    # negative n reverses the augmentation
    
    if n>0:
        # mirror x
        if (4 <= n <8) or (n>=12):
            pillar = np.flip(pillar, axis=0)
        
        # mirror z
        if n >= 8:
            pillar = np.flip(pillar, axis=2)

        # rotate
        pillar=np.rot90(pillar, k=n%4, axes=(0,1))

    elif n<0:
        n *= -1

        # Reverse rotate
        pillar = np.rot90(pillar, k=- (n % 4), axes=(0, 1))

        # Reverse mirror z
        if n >= 8:
            pillar = np.flip(pillar, axis=2)

        # Reverse mirror x
        if (4 <= n < 8) or (n >= 12):
            pillar = np.flip(pillar, axis=0)
    
    return pillar

def augment_n_flip_parallel(pillars, n):
    # does data augmentation for an array of pillars. For positive n, n in range [0, 15].
    # negative n reverses the augmentation
    
    if n>0:
        # mirror x
        if (4 <= n <8) or (n>=12):
            pillars = np.flip(pillars, axis=1)
        
        # mirror z
        if n >= 8:
            pillars = np.flip(pillars, axis=3)

        # rotate
        pillars=np.rot90(pillars, k=n%4, axes=(1,2))

    elif n<0:
        n *= -1

        # Reverse rotate
        pillars = np.rot90(pillars, k=- (n % 4), axes=(1, 2))

        # Reverse mirror z
        if n >= 8:
            pillars = np.flip(pillars, axis=3)

        # Reverse mirror x
        if (4 <= n < 8) or (n >= 12):
            pillars = np.flip(pillars, axis=1)
    
    return pillars

