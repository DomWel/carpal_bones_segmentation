import numpy as np
import tensorflow as tf
from PIL import Image
from helper_functions import preprocessImagePIL, createImageWithMaskLabels
import random
import config  # only needed if images of the dataloader output are generated to test preprocessing functions

class DataGeneratorUNET_OHE2(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, img_src_directory, batch_size=32, dim=(512, 512), n_channels=1,
                 shuffle=True, n_classes=7, random_crop_coeff = None, autocontrast=True, padding=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.img_src_directory = img_src_directory
        self.random_crop_coeff = random_crop_coeff
        self.autocontrast = autocontrast
        self.padding = padding
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, self.n_classes+1))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            img_path = self.img_src_directory + "/" + ID 
            img_orig = Image.open(img_path).convert('RGB')

            if self.random_crop_coeff != None:
              random_crop_size = random.uniform(self.random_crop_coeff, 1.0)
              random_crop_pos_x = random.uniform(0, 1.0)
              random_crop_pos_y = random.uniform(0, 1.0) 
              random_crop_params = [random_crop_size, random_crop_pos_x, random_crop_pos_y]
            else: 
              random_crop_params = None
            
            img = preprocessImagePIL(img_orig, ID, convert_grayscale=True, 
                      dim=self.dim, random_crop_params=random_crop_params, 
                      autocontrast=self.autocontrast, padding=self.padding)
            
            #img_for_image = Image.fromarray(img[:,:,0]*255)
            X[i,] = img 

            one_hot = np.zeros((img.shape[0], img.shape[1], self.n_classes+1))
            background = np.ones((img.shape[0], img.shape[1]))

            mask_path = self.img_src_directory + "/" + self.labels[ID] + ".npy"
            mask_np = np.load(mask_path)
            
            for index in range(mask_np.shape[2]):
                img = Image.fromarray(mask_np[:,:,index])
                
                # Apply random crop
                img = preprocessImagePIL(img, ID, convert_grayscale=False, 
                      dim=(512,512), random_crop_params=random_crop_params, 
                      autocontrast=False, padding=self.padding, norm=False)
       
                one_hot[:, :, index+1] = img[:,:,0]
                background = background - img[:,:,0]
                background = np.where(background > 0, 1, 0)

            one_hot[:, :, 0] = background

            # Check datalaoder output visually
            #img = createImageWithMaskLabels(img_for_image, one_hot, config.others['color_list'])
            #img.save(config.dirs['results_path']+ "/" +str(i)+".png")
            y[i] = one_hot
    
        return X, y

