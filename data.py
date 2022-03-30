import numpy as np
import keras
import tensorflow as tf
from PIL import Image
from skimage.transform import resize
import config

class DataGeneratorUNET_OHE(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, img_src_directory, batch_size=32, dim=(512, 512), n_channels=1,
                 shuffle=True, n_classes=7):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.img_src_directory = img_src_directory
        
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
        y = np.empty((self.batch_size, *self.dim, self.n_classes))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            
            # Store sample
            #img_path = '/content/drive/MyDrive/BoneSegm/mask_labels_carpal_bones/' + ID
            img_path = self.img_src_directory + "/" + ID 
            img = Image.open(img_path).convert('L')
            
            newsize = self.dim
            img = img.resize(newsize)
            
            img = np.array(img) / 255
            
            img = np.expand_dims(img, 2)
            X[i,] = img 

            list_carpal_bones = ['Os scaphoideum', 'Os lunatum', 'Os trapezium', 'Os trapezoideum', \
                      'Os hamatum', 'Os capitatum', 'Os triquetrum']
                      
            one_hot = np.zeros((img.shape[0], img.shape[1], self.n_classes))
            background = np.ones((img.shape[0], img.shape[1]))

            # Store class
            for index, bone in enumerate(list_carpal_bones, start=1): 
                key = 'labels_dict_' + bone
                #img_path = '/content/drive/MyDrive/BoneSegm/mask_labels_carpal_bones/' + self.labels[key][ID]
                img_path = self.img_src_directory + "/" + self.labels[key][ID]
                img = Image.open(img_path).convert('L')
                img = img.resize(self.dim)
                img = np.array(img) / 255.0

                one_hot[:, :, index] = img

                background = background - img
                background = np.where(background > 0, 1, 0)


 
            one_hot[:, :, 0] = background
            y[i] = one_hot
    
        return X, y


class DataGeneratorUNET_OHE2(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, img_src_directory, batch_size=32, dim=(512, 512), n_channels=1,
                 shuffle=True, n_classes=7):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.img_src_directory = img_src_directory
        
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
            # img_path = '/content/drive/MyDrive/BoneSegm/mask_labels_carpal_bones/' + ID
            img_path = self.img_src_directory + "/" + ID 
            
            img_orig = Image.open(img_path).convert('RGB')
            img = Image.open(img_path).convert('L')
            
            newsize = self.dim
            img_resized = img.resize(newsize)
            
            img = np.array(img_resized) / 255
            
            img = np.expand_dims(img, 2)
            X[i,] = img 

            one_hot = np.zeros((img.shape[0], img.shape[1], self.n_classes+1))
            background = np.ones((img.shape[0], img.shape[1]))

            mask_path = self.img_src_directory + "/" + self.labels[ID] + ".npy"
            mask_np = np.load(mask_path)
            


            for index in range(mask_np.shape[2]):
                img = Image.fromarray(mask_np[:,:,index])
                img = img.resize(self.dim)
                img = np.array(img) * 1.00
                
                one_hot[:, :, index+1] = img
                background = background - img
                background = np.where(background > 0, 1, 0)

              #one_hot[:,:,i] = resize(mask_np[:,:,i-1], newsize) * 1.0
              #background = background - one_hot[:,:,i]
              #background = np.where(background > 0, 1, 0)

            
            #img = Image.fromarray(background * 255.0).convert('L')
            #img.save(config.dirs['image_results'] + "/" +  self.labels[ID] + ".png")
            # final_image = rescaled_image.astype(np.uint8)
            one_hot[:, :, 0] = background
            #createImageWithMaskLabels(img_resized, one_hot).save(config.dirs['image_results'] + "/" +  self.labels[ID] + ".png")

            y[i] = one_hot
    
        return X, y

def createImageWithMaskLabels(orig_img, masks_np):
  colors = [(255, 0, 0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255), (100, 200, 0), (230, 0, 60), (60,230,0), (250, 0, 0)]
  orig_img = orig_img.convert('RGB')
  for index in range(1):
    img_array = masks_np[:, :, index] * 255
    
    img = Image.fromarray(img_array * 0.3)
    img_RGB = img.convert('RGB')
    fusion_mask = img.convert('L')
    pixels = img_RGB.load()
    # Convert to corresponding color
    for i in range(img_RGB.size[0]): # for every pixel:
      for j in range(img_RGB.size[1]):
          red_value = pixels[i,j][0]
          if red_value > 50:
            pixels[i,j] =  colors[index]
    img_RGB.putalpha(fusion_mask)
    orig_img.paste(img_RGB, (0,0), img_RGB)
  return orig_img