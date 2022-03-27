import numpy as np
import keras
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, list_IDs_img_masks, batch_size=32, dim=(2,600,600), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.list_IDs_img_masks = list_IDs_img_masks
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            
            # Store sample
            img_path = self.list_IDs_img_masks[ID][0]
            coords = self.list_IDs_img_masks[ID][1]

            img = Image.open(img_path).convert('L')
            img = np.array(img)
            #img = np.expand_dims(img, axis=0)
            img_shape = img.shape
            coords[1] = img.shape[0] - coords[1]

            list_bones = ['Scapho', 'Lunatum', 'Trapezium', 'Capitatum', 'Trique', 'Trapezoideum', 'Pisiforme', 'Hamatum']
            max_val = np.amax(img)
            img = img * 1.0 / float(max_val)
                
            img[ int(coords[1])-5:int(coords[1])+5, int(coords[0])-5:int(coords[0])+5 ] = 1000.0
            
            #img = img[ int(coords[1])-20:int(coords[1])+20, int(coords[0])-20:int(coords[0])+20 ]
            
            #Padding
            #target_input_size = np.zeros([60, 60])
            #target_input_size[:img.shape[0],:img.shape[1]] = img


            img = Image.fromarray(img)
            newsize = (300, 300)
            img = img.resize(newsize)
            #img.save("/content/drive/MyDrive/BoneSegm/test_images/test" + list_bones[self.labels[ID]] + ".jpeg")
            img = np.array(img)
            
            #target_input_size_tensor = tf.convert_to_tensor(target_input_size)

            img = np.expand_dims(img, axis=2)
     
          
         
            X[i,] = img 


            # Store class
            y[i] = self.labels[ID]
            

        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)



class MNISTDataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, x_train, labels, batch_size=32, dim=(2,600,600), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.x_train = x_train
        #self.list_IDs = list_IDs
        #self.list_IDs_img_masks = list_IDs_img_masks
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.x_train) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Find list of IDs
        X_ = [self.x_train[k] for k in indexes]

        for i, x in enumerate(X_):
            x = Image.fromarray(x)
            newsize = (50, 50)
            x = x.resize(newsize)
            #img.save("/content/drive/MyDrive/BoneSegm/test_images/test" + list_bones[self.labels[ID]] + ".jpeg")
            x = np.array(x)
            x = np.expand_dims(x, axis=2)
            X[i,] = x

        y_list = [self.labels[k] for k in indexes]
        # Store class
        for i, elem in enumerate(y_list):
          y[i] = elem 
        
        


        
        y = tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
        
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.x_train))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            
            # Store sample
            img_path = self.list_IDs_img_masks[ID][0]
            coords = self.list_IDs_img_masks[ID][1]

            img = Image.open(img_path).convert('L')
            img = np.array(img)
            #img = np.expand_dims(img, axis=0)
            img_shape = img.shape
            coords[1] = img.shape[0] - coords[1]

            list_bones = ['Scapho', 'Lunatum', 'Trapezium', 'Capitatum', 'Trique', 'Trapezoideum', 'Pisiforme', 'Hamatum']
            max_val = np.amax(img)
            img = img * 1.0 / float(max_val)
                
            img[ int(coords[1])-5:int(coords[1])+5, int(coords[0])-5:int(coords[0])+5 ] = 1000.0
            
            #img = img[ int(coords[1])-20:int(coords[1])+20, int(coords[0])-20:int(coords[0])+20 ]
            
            #Padding
            #target_input_size = np.zeros([60, 60])
            #target_input_size[:img.shape[0],:img.shape[1]] = img


            img = Image.fromarray(img)
            newsize = (300, 300)
            img = img.resize(newsize)
            #img.save("/content/drive/MyDrive/BoneSegm/test_images/test" + list_bones[self.labels[ID]] + ".jpeg")
            img = np.array(img)
            
            #target_input_size_tensor = tf.convert_to_tensor(target_input_size)

            img = np.expand_dims(img, axis=2)
     
          
         
            X[i,] = img 


            # Store class
            y[i] = self.labels[ID]
            

        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)

class DataGeneratorUNET(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(2,600,600), n_channels=1,
                 shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        
        self.n_channels = n_channels
        
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, 1))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            
            # Store sample
            img_path = '/content/drive/MyDrive/BoneSegm/Mask_labels_lunatum/' + ID
            
            img = Image.open(img_path).convert('L')
            
            newsize = self.dim
            img = img.resize(newsize)
            
            img = np.array(img) / 255
            
            #img = np.expand_dims(img, 2)
            X[i,] = img 

            # Store class
            img_path_label = '/content/drive/MyDrive/BoneSegm/Mask_labels_lunatum/' + self.labels[ID]
            
            img = Image.open(img_path_label).convert('L')
            
            
            

            
            img = img.resize(self.dim)
            
            img = np.array(img) / 255.0
            #img.save("/content/drive/MyDrive/BoneSegm/test_images/test" + str(i) + ".jpeg")
            #print(np.amax(img), np.amin(img))
            img = np.expand_dims(img, 2)
            
            

            y[i] = img
            
        
        

        return X, y


class DataGeneratorUNET_OHE(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(2,600,600), n_channels=1,
                 shuffle=True, n_classes=7):
        print('Init of data loader')
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.n_channels = n_channels
        
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, self.n_classes))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            
            # Store sample
            img_path = '/content/drive/MyDrive/BoneSegm/mask_labels_carpal_bones/' + ID
            
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
                img_path = '/content/drive/MyDrive/BoneSegm/mask_labels_carpal_bones/' + self.labels[key][ID]
                img = Image.open(img_path).convert('L')
                img = img.resize(self.dim)
                img = np.array(img) / 255.0

                one_hot[:, :, index] = img

                background = background - img
                background = np.where(background > 0, 1, 0)

            one_hot[:, :, 0] = background

            """
            img_path_label = '/content/drive/MyDrive/BoneSegm/training_data_scapho_lun/' + self.labels['labels_dict_Os lunatum'][ID]
            img_path_label2 = '/content/drive/MyDrive/BoneSegm/training_data_scapho_lun/' + self.labels['labels_dict_Os scaphoideum'][ID]
            
            img = Image.open(img_path_label).convert('L')
            img2 = Image.open(img_path_label2).convert('L')
            
            

            
            img = img.resize(self.dim)
            img2 = img2.resize(self.dim)

            img = np.array(img) / 255.0
            img2 = np.array(img2) / 255.0
            #img.save("/content/drive/MyDrive/BoneSegm/test_images/test" + str(i) + ".jpeg")
            #print(np.amax(img), np.amin(img))
            #img = np.expand_dims(img, 2)
            
            n_classes = 3
            one_hot = np.zeros((img.shape[0], img.shape[1], n_classes))
            

            background = 1 - img - img2
            
            background = np.where(background > 0, 1, 0)
            

            
           
            
           

            one_hot[:, :, 0] = background
            one_hot[:, :, 1] = img
            one_hot[:, :, 2] = img2

   

            #y[i] = img
            """

            y[i] = one_hot
            
            
        
        

        return X, y