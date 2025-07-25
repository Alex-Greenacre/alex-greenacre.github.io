---
layout: post
title:  "Identifying Cancer Cells - Uni Machine Learning Report"
date:   2025-05-29 11:15:00 +0100
categories: report
---
# Alex Greenacre - Msc ML Assignment
 

## 1: Data loading

The goal of this Assignment is to explore a dataset containing images of cancerous and none cancerous cells with the purpose to train models to classify an image between these two classifications. To do this two models will be built, trained and compared against  


```python
!pip install keras-tuner
#Machine learning libaries
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import keras_tuner as kt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve ,PrecisionRecallDisplay,auc,roc_auc_score
from sklearn.svm import LinearSVC

#Data mangement and visualisation libaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random as rand


#Libaries used for data loading
from google.colab import drive
import os

import warnings
```

    Requirement already satisfied: keras-tuner in /usr/local/lib/python3.10/dist-packages (1.4.7)
    Requirement already satisfied: keras in /usr/local/lib/python3.10/dist-packages (from keras-tuner) (2.15.0)
    Requirement already satisfied: packaging in /usr/local/lib/python3.10/dist-packages (from keras-tuner) (24.0)
    Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from keras-tuner) (2.31.0)
    Requirement already satisfied: kt-legacy in /usr/local/lib/python3.10/dist-packages (from keras-tuner) (1.0.5)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->keras-tuner) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->keras-tuner) (3.7)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->keras-tuner) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->keras-tuner) (2024.2.2)



```python
#Load drive
drive.mount('/content/drive',force_remount=True)
#Navigate to data
os.chdir('/content/drive/My Drive/Colab Notebooks/Assignment 2/Data')
#Confirm location
!pwd
```

    Mounted at /content/drive
    /content/drive/My Drive/Colab Notebooks/Assignment 2/Data



```python
#Dataset
folderName = "./BHI_Dataset/"
images = np.load(folderName+'X.npy')
labels= np.load(folderName+'Y.npy')
```

## 2: Data Exploration

Before any model can be built the dataset needs to be explored and understood to do this the datasets dimensions, features and distributions will be considered


### 2.1: Dataset overview


Given the initial exploration carried out shows that the dataset from the X.py file contains a numpy array representing over 5000 images at 50x50 with 3  dimensions for rgb resolution furthermore the accompanying Y.py file contains the labelling of each image using a binary classification of 0 and 1, an assumption will be made that 0 classifies as safe and 1 classifying the data as a cancerous image .    


```python
print("Image shape: ",images.shape,"\nLabel shape: ",labels.shape)
print("Label values: ",np.unique(labels))
```

    Image shape:  (5547, 50, 50, 3) 
    Label shape:  (5547,)
    Label values:  [0 1]


If the dataset has a uneven distribution of one classification it could lead to a biased in our models whilst training making the model favour one classification over the other, to avoid this the dataset needs to have a balanced amount of classification



```python
#Get a overview of the data
totalCellCount = labels.size
sCellCount = labels.tolist().count(0)
cCellCount = labels.tolist().count(1)
print("Safe cells: ",sCellCount,"\nCancerous cells: ",cCellCount)
print("Cell split\nSafe:",sCellCount/(totalCellCount/100),"%",
      "\nCancerous:",cCellCount/(totalCellCount/100),"%")


#Check shape of each classification
safeArray = []
cancerousArray = []
for i in range(0,labels.size,1):
  if labels[i] == 0:
    safeArray.append(images[i])
  else:
    cancerousArray.append(images[i])
safeArray = np.array(safeArray)
cancerousArray = np.array(cancerousArray)
print(safeArray.shape,'\n',cancerousArray.shape)
```

    Safe cells:  2759 
    Cancerous cells:  2788
    Cell split
    Safe: 49.73859744005769 % 
    Cancerous: 50.26140255994231 %
    (2759, 50, 50, 3) 
     (2788, 50, 50, 3)


As shown the exploration the split between classification 0 and 1 is close to even meaning that additional no balancing of data needs to occur.

As well as categorisation, the datasets 5000 images may not be enough for neural networks and it could be worth exploring additional data generation in pre processing to increase the models accuracy  

### 2.2: Visual exploration

To gain further understanding of the data 5 random safe and cancerous images were loaded and displayed, looking at these images shows that there is very little to distinguish each category when considering the images shape, although colour could be a defining feature with cancerous images often containing darker colours than the safe images.



```python
rand.seed(1)
#Randomly select 5 images of each classification
index_of_safe = [i for i in range(0,len(labels)) if labels[i]==0]
index_of_cancerous = [i for i in range(0,len(labels)) if labels[i]==1]
selected_index_safe = rand.sample(index_of_safe,5)
selected_index_cancerous = rand.sample(index_of_cancerous ,5)
#Display the selected images
fig,ax =plt.subplots(2,5)
ax[0,0].imshow(images[selected_index_safe[0]])
ax[0,1].imshow(images[selected_index_safe[1]])
ax[0,2].imshow(images[selected_index_safe[2]])
ax[0,3].imshow(images[selected_index_safe[3]])
ax[0,4].imshow(images[selected_index_safe[4]])
ax[1,0].imshow(images[selected_index_cancerous[0]])
ax[1,1].imshow(images[selected_index_cancerous[1]])
ax[1,2].imshow(images[selected_index_cancerous[2]])
ax[1,3].imshow(images[selected_index_cancerous[3]])
ax[1,4].imshow(images[selected_index_cancerous[4]])
for i in range(0,2):
  for x in range(0,5):
    ax[i,x].axis('off')
plt.subplots_adjust(wspace=0.1,hspace=0.1)
plt.show()
```


    
![png](/assets/report_assets/2025-06-18-ML_Assignment/output_17_0.png)
    


### 2.3: Colour Distribution

To explore the colour differences further the mean and standard deviation (std) for each colour dimension was taken for all images, safe images and cancerous images.


```python
RGB_dim_one = images[:,:,:,0]
RGB_dim_two = images[:,:,:,1]
RGB_dim_three=images[:,:,:,2]
print('Total          Mean               STD',
      '\ndim 1:',np.mean(RGB_dim_one),'|',np.std(RGB_dim_one),
      '\ndim 2:',np.mean(RGB_dim_two),'|',np.std(RGB_dim_two),
      '\ndim 3:',np.mean(RGB_dim_three),'|',np.std(RGB_dim_three))

safe_images = np.array([images[i] for i in index_of_safe])
cancerous_images = np.array([images[i] for i in index_of_cancerous])
safe_RGB_dim_one = safe_images[:,:,:,0]
safe_RGB_dim_two = safe_images[:,:,:,1]
safe_RGB_dim_three = safe_images[:,:,:,2]
print('\nSafe\ndim 1:',np.mean(safe_RGB_dim_one),'|',np.std(safe_RGB_dim_one),
      '\ndim 2:',np.mean(safe_RGB_dim_two),'|',np.std(safe_RGB_dim_two),
      '\ndim 3:',np.mean(safe_RGB_dim_three),'|',np.std(safe_RGB_dim_three))
cancerous_RGB_dim_one = cancerous_images[:,:,:,0]
cancerous_RGB_dim_two = cancerous_images[:,:,:,1]
cancerous_RGB_dim_three=cancerous_images[:,:,:,2]
print('\nCancerous\ndim 1:',np.mean(cancerous_RGB_dim_one),'|',np.std(cancerous_RGB_dim_one),
      '\ndim 2:',np.mean(cancerous_RGB_dim_two),'|',np.std(cancerous_RGB_dim_two),
      '\ndim 3:',np.mean(cancerous_RGB_dim_three),'|',np.std(cancerous_RGB_dim_three))


```

    Total          Mean               STD 
    dim 1: 205.79039149089598 | 36.29435804614739 
    dim 2: 161.866669983775 | 53.94012334259874 
    dim 3: 187.4441728501893 | 38.69085273296819
    
    Safe
    dim 1: 218.09366698079015 | 30.493448288311576 
    dim 2: 177.3471375135919 | 52.984002213355375 
    dim 3: 197.3947237404857 | 38.7993928900707
    
    Cancerous
    dim 1: 193.6150912482066 | 37.45192271406365 
    dim 2: 146.54722596843615 | 50.387184959654256 
    dim 3: 177.59712482065999 | 35.96819235343645


the results of this proves that the observation made in visual exploration as cancerous images had a lower mean and standard deviation in almost all colour dimensions when compared to the safe images with standard deviation only being a higher value in the first dimension. This shows that the cancerous images are often darker, this means that colour dimension reduction could be considered when pre processing to further expose this feature in the images.  

## 3: Pre Proccessing

### 3.1: Dimension Reduction With AutoEncoder

To improve the performance and speed of the models optimisations can be made to the dataset, as the images are already small the focus will instead be on reducing the dimensions of the images colours reducing the three RGB dimensions of 50 by 50 into a single grey scale dimension. This will not only decrease the size of the images but it should also help with feature extraction, with features in the cancerous classification displaying a higher contrast on certain features these should be more visible when converted to greyscale    


```python
#Creates the input layer for the autoencoder, as the encoder dimensions will take the current image shape image[1:] can be used
input = layers.Input(shape = images.shape[1:])
#Dimension expanded from 3 to 128
encoder = layers.Dense(128)(input)
encoder = layers.BatchNormalization()(encoder)
encoder = layers.LeakyReLU()(encoder)
#Dimensions reduced
encoder = layers.Dense(64)(encoder)
encoder = layers.BatchNormalization()(encoder)
encoder = layers.LeakyReLU()(encoder)
#Dimensions reduced
encoder = layers.Dense(16)(encoder)
encoder = layers.BatchNormalization()(encoder)
encoder = layers.LeakyReLU()(encoder)
#Dimesnion reduced to 1
bottleneck = layers.Dense(1)(encoder)

```


```python
decoder_location = './autoencoder/decoder_weights.h5'
if os.path.exists(decoder_location):
  model = models.load_model(decoder_location)
  print('Decoder loaded')
else:
  model = models.Model(inputs=input,outputs=bottleneck)
  model.compile(optimizer='adam',loss='mse')
  model.save(decoder_location)
decoded_data = model.predict(images)
```

    /usr/local/lib/python3.10/dist-packages/keras/src/engine/training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
      saving_api.save_model(


    174/174 [==============================] - 46s 263ms/step



```python
fig,ax =plt.subplots(2,5)
ax[0,0].imshow(decoded_data[selected_index_safe[0]],cmap='gray')
ax[0,1].imshow(decoded_data[selected_index_safe[1]],cmap='gray')
ax[0,2].imshow(decoded_data[selected_index_safe[2]],cmap='gray')
ax[0,3].imshow(decoded_data[selected_index_safe[3]],cmap='gray')
ax[0,4].imshow(decoded_data[selected_index_safe[4]],cmap='gray')
ax[1,0].imshow(decoded_data[selected_index_cancerous[0]],cmap='gray')
ax[1,1].imshow(decoded_data[selected_index_cancerous[1]],cmap='gray')
ax[1,2].imshow(decoded_data[selected_index_cancerous[2]],cmap='gray')
ax[1,3].imshow(decoded_data[selected_index_cancerous[3]],cmap='gray')
ax[1,4].imshow(decoded_data[selected_index_cancerous[4]],cmap='gray')
for i in range(0,2):
  for x in range(0,5):
    ax[i,x].axis('off')
plt.subplots_adjust(wspace=0.1,hspace=0.1)
plt.show()
```


    
![png](/assets/report_assets/2025-06-18-ML_Assignment/output_27_0.png)
    


Having reduced the colour dimensions with the autoencoder the results show that the images shape still appears the same as pre dimension reduction, furthermore the cancerous images appear to have a higher colour contrast in certain places indicating that some feature extraction has occurred   

## 3.2: Creating New Data

As the sample size in of the dataset is quite small it could leads the model to overfit data to address this the images in the model will be flipped 3 times (giving 4 instances of each image at 90 degrees) as visual inspection shows that orientation does not matter in this dataset the results should produce a dataset with 4x the images than the original reducing the chance of overfitting when training the model.


```python
flipped_hor_data = np.array([np.fliplr(x) for x in decoded_data])
flipped_ver_data = np.array([np.flipud(x) for x in decoded_data])
flipped_hor_ver_data = np.array([np.flipud(x) for x in flipped_hor_data])
fit_data = np.concatenate((decoded_data,flipped_hor_data,flipped_ver_data,flipped_hor_ver_data))
fit_labels = np.concatenate((labels,labels,labels,labels))
fit_data.shape
```




    (22188, 50, 50, 1)




```python
#Checks images flipped correctly
fig,ax =plt.subplots(2,2)
ax[0,0].imshow(fit_data[0], cmap='gray')
ax[0,1].imshow(fit_data[decoded_data.shape[0]], cmap='gray')
ax[1,0].imshow(fit_data[decoded_data.shape[0]*2], cmap='gray')
ax[1,1].imshow(fit_data[decoded_data.shape[0]*3], cmap='gray')
plt.show()
#checks labels have been concatenated correctly
print('Labels: ',fit_labels[0],fit_labels[decoded_data.shape[0]],fit_labels[decoded_data.shape[0]*2],fit_labels[decoded_data.shape[0]*3])
```


    
![png](/assets/report_assets/2025-06-18-ML_Assignment/output_32_0.png)
    


    Labels:  0 0 0 0


### 3.3: Train Test Split


```python
#Data for CNN
X_train,X_test,y_train,y_test = train_test_split(fit_data,fit_labels,test_size=0.3,random_state=1)
#Get the parameters used for hyperparameter testing, as only test set is needed train vals are set to a unused var
#Set the size to 0.15 to set the size to be roughly 10% of the inital dataset
X_a,X_hp,y_a,y_hp = train_test_split(X_train,y_train,test_size=0.15,random_state=1)
print ('Train data shape X: ',X_train.shape,' Y: ', y_train.shape,
       '\nTest data shape X: ',X_test.shape,' Y: ', y_test.shape,
       '\nHyper parameter data shape X: ',X_hp.shape,' Y: ', y_hp.shape)

#Data for svm
#Flatten array into a 2d array
flattend_data = decoded_data.reshape(decoded_data.shape[0],-2)
#Data will not use flipped images
X_train_svm,X_test_svm,y_train_svm,y_test_svm = train_test_split(flattend_data,labels,test_size=0.3,random_state=1)
X_a,X_hp_svm,y_a,y_hp_svm = train_test_split(X_train_svm,y_train_svm,test_size=0.15,random_state=1)
print ('Train data shape X: ',X_train_svm.shape,' Y: ', y_train_svm.shape,
       '\nTest data shape X: ',X_test_svm.shape,' Y: ', y_test_svm.shape,
       '\nHyper parameter data shape X: ',X_hp_svm.shape,' Y: ', y_hp_svm.shape)
```

    Train data shape X:  (15531, 50, 50, 1)  Y:  (15531,) 
    Test data shape X:  (6657, 50, 50, 1)  Y:  (6657,) 
    Hyper parameter data shape X:  (2330, 50, 50, 1)  Y:  (2330,)
    Train data shape X:  (3882, 2500)  Y:  (3882,) 
    Test data shape X:  (1665, 2500)  Y:  (1665,) 
    Hyper parameter data shape X:  (583, 2500)  Y:  (583,)


## 4: Model 1 - CNN

### 4.1: Design

The first model used to classify the dataset is a Convolutional neural network,
This was chosen due to CNN popularity for image classification of medical images, this can be best seen in Ghaffari (2019) who when investigating the Brats challenges, a challenge surrounding ml models for brain tumour identification, finds that CNN was involved in numerous entries of the BRATS challenge from 2014 to 2018. This further shows why CNN is a strong choice for this scenario    

The architecture of the model will consist of 4 steps the first will be a Conv2d layer that will take the input shape of the pre-processed images and process it through the first layer, after the input layer a batch normalisation layer is used to ensure that the data is normalised before continuing through the network, the final layer in the first step is to reduce the size of the shape using MaxPooling reducing it by half. The Conv2d and MaxPooling layers are then repeated twice more in the second and third steps, this gives the model 3 steps where the model views the data with Conv2d and downsamples with MaxPooling giving a dimension reduction of 50,25,13 and 7. The final part of step 3 is a dropout layer this layer is introduced to limit overfitting by removing half of the data.  

With 3 of 4 steps executed the final step is to flatten the remaining shape and pass it into a Dense layer with a ReLu activation, this will then pass the values to a final output layer which is a 1 unit Dense layer with a Sigmoid activation function, this function was used as the final output as it will produce a value between 0 and 1 which can be interpreted to show the models confidence level showing the percentage of confidence the model has in classifying the data with and output of <0.5 being 1 and >0.5 being 0.              



```python
'''
As the model is used after fininding the hyper parameters
  it needs to be split up the model into segments allowing for the model to be ran
  seperatly once the parameters have been found
To do this the model is split into 3 functions
  tune_cnn_hp - sets the hyper parameters
  build_cnn_model - takes variables for the set parameters and builds the model
  tune_cnn_model - manages both functions passing in the returned values into the model
'''
#sets hyper parameters
def tune_cnn_hp(hp):
  #Hyper parameter values
  #Set to be between 1 and 3 times the images size 50,100,150
  layer_one_filter = hp.Int('layeronefilter',min_value=10,max_value=100,step=10)
  #as the pooling will half the size this will be 25,50,75
  layer_two_filter = hp.Int('layertwofilter',min_value=10,max_value=50,step=10)
  layer_three_filter = hp.Int('layerthreefilter',min_value=10,max_value=50,step=10)
  dense_unit = hp.Int('denseunit',min_value=64,max_value=512,step=64)
  #Set the learning rate to be 0.01,0.001 or 0.0001
  hp_learning_rate = hp.Choice('learning_rate',values=[1e-1,1e-2,1e-3])
  use_dropout = hp.Boolean('dropout')
  return (layer_one_filter,layer_two_filter,layer_three_filter,dense_unit,use_dropout,hp_learning_rate)

#Builds model
def build_cnn_model(layer_one_filter,layer_two_filter,layer_three_filter,dense_unit,use_dropout,hp_learning_rate):
  model = models.Sequential()
  model.add(layers.Conv2D(filters=layer_one_filter,kernel_size=(4,4),activation='relu',input_shape=(50,50,1)))
  model.add(layers.BatchNormalization())
  model.add(layers.MaxPooling2D((2,2)))
  model.add(layers.Conv2D(filters=layer_two_filter,kernel_size=(2,2),activation='relu'))
  model.add(layers.MaxPooling2D((2,2)))
  model.add(layers.Conv2D(filters=layer_three_filter,kernel_size=(2,2),activation='relu'))
  model.add(layers.MaxPooling2D((2,2)))
    #Dropout to avoid overfitting
    #Use hyper params to check if this is required/model at risk of overfitting
  if use_dropout:
    model.add(layers.Dropout(0.5))
  model.add(layers.Flatten())
  model.add(layers.Dense(units=dense_unit, activation='relu'))
  #As model is classifing binary (yes or no) sigmoid is used
  #This will produce a percent of how certain the model thinks its a cancerous cell
  model.add(layers.Dense(1, activation='sigmoid'))

  model.compile(optimizers.Adam(learning_rate=hp_learning_rate),loss='binary_crossentropy',metrics=['accuracy'])
  return model
#returns tuned cnn model
#* used for unpacking vars from tune_cnn_hp
def tune_cnn_model(hp):
  return build_cnn_model(*tune_cnn_hp(hp))
```

With the model design optimisation needs to be set to allow the model to monitor its performance and learn to produce better results, to do this the optimizer used will be the adam optimizer, as the model will output between 0 and 1 the loss will be messured using the  binary_crossentropy setting further more as well as loss the accuracy will also be measured by the model to help indicate the performance of the model


### 4.2: Hyper Parameter Tuning

To ensure that the model will run as accurate as possible hyper parameter tuning is used this is used to ensure that each layer has the best performing amount of filters, compare different learning rate will be explored to ensure the best possible learning steps for the model.    



```python
#Create tuner
#If tuner already exists the tuner is loaded from the file
tuner = kt.Hyperband(tune_cnn_model,objective='val_accuracy',max_epochs=10,directory="./cnn_hp_tune_test/",project_name="CNN_Tuned_Params")

tuner.search(X_hp,y_hp,epochs=10,validation_split=0.2)
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best layer one: ",best_hp.get('layeronefilter'),
      "\nBest layer two: ",best_hp.get('layertwofilter'),
      "\nBest layer two: ",best_hp.get('layerthreefilter'),
      "\nBest dense layer: ",best_hp.get('denseunit'),
      "\nDropout: ",best_hp.get('dropout'),
      "\nLearning rate:",best_hp.get('learning_rate'))
```

    Trial 30 Complete [00h 01m 26s]
    val_accuracy: 0.6115880012512207
    
    Best val_accuracy So Far: 0.6952789425849915
    Total elapsed time: 00h 21m 14s
    Best layer one:  10 
    Best layer two:  20 
    Best layer two:  10 
    Best dense layer:  448 
    Dropout:  True 
    Learning rate: 0.001


### 4.3: Build & Run Of Model

The model will be ran against the test data for 20 epochs using a batch size of 400


```python
#Build model with the optimised hyperparamter values
cnnModel = build_cnn_model(best_hp.get('layeronefilter'),
                           best_hp.get('layertwofilter'),
                           best_hp.get('layerthreefilter'),
                           best_hp.get('denseunit'),
                           best_hp.get('dropout'),
                           best_hp.get('learning_rate'))
cnnModel.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv2d_3 (Conv2D)           (None, 47, 47, 10)        170       
                                                                     
     batch_normalization_1 (Bat  (None, 47, 47, 10)        40        
     chNormalization)                                                
                                                                     
     max_pooling2d_3 (MaxPoolin  (None, 23, 23, 10)        0         
     g2D)                                                            
                                                                     
     conv2d_4 (Conv2D)           (None, 22, 22, 20)        820       
                                                                     
     max_pooling2d_4 (MaxPoolin  (None, 11, 11, 20)        0         
     g2D)                                                            
                                                                     
     conv2d_5 (Conv2D)           (None, 10, 10, 10)        810       
                                                                     
     max_pooling2d_5 (MaxPoolin  (None, 5, 5, 10)          0         
     g2D)                                                            
                                                                     
     dropout (Dropout)           (None, 5, 5, 10)          0         
                                                                     
     flatten_1 (Flatten)         (None, 250)               0         
                                                                     
     dense_2 (Dense)             (None, 448)               112448    
                                                                     
     dense_3 (Dense)             (None, 1)                 449       
                                                                     
    =================================================================
    Total params: 114737 (448.19 KB)
    Trainable params: 114717 (448.11 KB)
    Non-trainable params: 20 (80.00 Byte)
    _________________________________________________________________



```python
cnnResults = cnnModel.fit(X_train,y_train,epochs=20,batch_size=400,validation_data=(X_test,y_test))
cnnPredicted = cnnModel.predict(X_test)
```

    Epoch 1/20
    39/39 [==============================] - 29s 726ms/step - loss: 0.6425 - accuracy: 0.6582 - val_loss: 3.6921 - val_accuracy: 0.4993
    Epoch 2/20
    39/39 [==============================] - 28s 725ms/step - loss: 0.5641 - accuracy: 0.7278 - val_loss: 2.1734 - val_accuracy: 0.4993
    Epoch 3/20
    39/39 [==============================] - 28s 725ms/step - loss: 0.5537 - accuracy: 0.7332 - val_loss: 1.3196 - val_accuracy: 0.5004
    Epoch 4/20
    39/39 [==============================] - 28s 725ms/step - loss: 0.5406 - accuracy: 0.7394 - val_loss: 1.0713 - val_accuracy: 0.5112
    Epoch 5/20
    39/39 [==============================] - 28s 726ms/step - loss: 0.5371 - accuracy: 0.7443 - val_loss: 0.8867 - val_accuracy: 0.5469
    Epoch 6/20
    39/39 [==============================] - 29s 741ms/step - loss: 0.5339 - accuracy: 0.7472 - val_loss: 0.7395 - val_accuracy: 0.6087
    Epoch 7/20
    39/39 [==============================] - 26s 659ms/step - loss: 0.5316 - accuracy: 0.7459 - val_loss: 0.7525 - val_accuracy: 0.6590
    Epoch 8/20
    39/39 [==============================] - 28s 708ms/step - loss: 0.5241 - accuracy: 0.7510 - val_loss: 0.7035 - val_accuracy: 0.6649
    Epoch 9/20
    39/39 [==============================] - 31s 789ms/step - loss: 0.5271 - accuracy: 0.7485 - val_loss: 0.7902 - val_accuracy: 0.6378
    Epoch 10/20
    39/39 [==============================] - 31s 794ms/step - loss: 0.5216 - accuracy: 0.7504 - val_loss: 0.6089 - val_accuracy: 0.7122
    Epoch 11/20
    39/39 [==============================] - 31s 794ms/step - loss: 0.5156 - accuracy: 0.7533 - val_loss: 0.6250 - val_accuracy: 0.7084
    Epoch 12/20
    39/39 [==============================] - 28s 734ms/step - loss: 0.5223 - accuracy: 0.7522 - val_loss: 0.5808 - val_accuracy: 0.7295
    Epoch 13/20
    39/39 [==============================] - 30s 766ms/step - loss: 0.5177 - accuracy: 0.7533 - val_loss: 0.5416 - val_accuracy: 0.7458
    Epoch 14/20
    39/39 [==============================] - 29s 736ms/step - loss: 0.5123 - accuracy: 0.7566 - val_loss: 0.5538 - val_accuracy: 0.7394
    Epoch 15/20
    39/39 [==============================] - 29s 740ms/step - loss: 0.5129 - accuracy: 0.7552 - val_loss: 0.5365 - val_accuracy: 0.7427
    Epoch 16/20
    39/39 [==============================] - 29s 740ms/step - loss: 0.5101 - accuracy: 0.7568 - val_loss: 0.5303 - val_accuracy: 0.7490
    Epoch 17/20
    39/39 [==============================] - 29s 743ms/step - loss: 0.5079 - accuracy: 0.7596 - val_loss: 0.5428 - val_accuracy: 0.7403
    Epoch 18/20
    39/39 [==============================] - 27s 694ms/step - loss: 0.5045 - accuracy: 0.7589 - val_loss: 0.5578 - val_accuracy: 0.7335
    Epoch 19/20
    39/39 [==============================] - 31s 779ms/step - loss: 0.5024 - accuracy: 0.7644 - val_loss: 0.5219 - val_accuracy: 0.7511
    Epoch 20/20
    39/39 [==============================] - 28s 722ms/step - loss: 0.5028 - accuracy: 0.7647 - val_loss: 0.5078 - val_accuracy: 0.7563
    209/209 [==============================] - 4s 20ms/step


### 4.4: Results And Evaluation

To evaluate the model 4 key areas will be investigated, the accuracy and loss when training the model, a confusion matrix and the precision recall and f1 score it produces, a ROC curve and a precision recall curve.


```python
fig,ax = plt.subplots(2,1)
ax[0].plot(cnnResults.history['loss'],label='loss' )
ax[0].plot(cnnResults.history['val_loss'],label='val loss')
ax[1].plot(cnnResults.history['accuracy'],label='accuracy')
ax[1].plot(cnnResults.history['val_accuracy'],label='val accuracy')
ax[0].set_title('Loss')
ax[1].set_title('Accuracy')
ax[0].legend()
ax[1].legend()
plt.subplots_adjust(wspace=0.1,hspace=0.3)
plt.show()
```


    
![png](/assets/report_assets/2025-06-18-ML_Assignment/output_49_0.png)
    


Although the model shows a good learning progress of the model the lack of stabilisation in the accuracy and loss shows that the model may be starting to experience over fitting in the later epochs
   


```python
cnnPredictedClassification = [0 if x<0.5 else 1 for x in cnnPredicted]
matrix = confusion_matrix(y_test, cnnPredictedClassification)
sns.heatmap(matrix, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=['true', 'false'], yticklabels=['true', 'false'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
recall = matrix[0][0]/(matrix[0][0]+matrix[0][1])
precision = matrix[0][0]/(matrix[0][0]+matrix[1][0])
fscore = 2*((precision*recall)/(precision+recall))
plt.title('CNN Metrics')
#Display metrics along with the confusion matrix
plt.text(0.7,2.5,'Metrics')
plt.text(0.5,2.7,'Accuracy: '+str((matrix[0][0]+matrix[1][-1])/(matrix[0][0]+matrix[0][-1]+matrix[1][0]+matrix[1][-1])))
plt.text(0.5,2.8,'Recall: '+str(recall))
plt.text(0.5,2.9,'Precision: '+str(precision))
plt.text(0.5,3.0,'fscore: '+str(fscore))

```




    Text(0.5, 3.0, 'fscore: 0.7651997683844818')




    
![png](/assets/report_assets/2025-06-18-ML_Assignment/output_51_1.png)
    


The confusion matrix shows some good results with large portions of the data being true positive or negative, taking this data a accuracy of over 75% a recall of 0.79, a precision of 0.6 and a f1 score of 0.76 can be attained.


```python
fig, ax = plt.subplots(figsize=(8,8),nrows=2,ncols=1)
#ROC curve
false_pos_rate,true_pos_rate,t = roc_curve(y_test,cnnPredicted)
ax[0].plot(false_pos_rate,true_pos_rate)
ax[0].set_title('ROC Curve' )
ax[0].text(0,-0.35,'RoC area under curve score:'+str(roc_auc_score(y_test,cnnPredicted)))
ax[0].set_xlabel('false positvie rate')
ax[0].set_ylabel('true postitive rate')

#Presicion-Recall curve
#Plot the curve
plot = PrecisionRecallDisplay.from_predictions(y_test,cnnPredicted,ax=ax[1])
#Get the precision and recall for the auc calculation
precision, recall, t= precision_recall_curve(np.array(y_test),np.array(cnnPredicted))
pr_auc = auc(recall,precision)
ax[1].set_title('Presicion-Recall Curve' )
ax[1].text(0,0.35,'Presicion-Recall area under curve score: '+str(pr_auc))
plt.subplots_adjust(wspace=0.1,hspace=0.5)
plt.show()
```


    
![png](/assets/report_assets/2025-06-18-ML_Assignment/output_53_0.png)
    


#### Roc curve
Roc curve is used to show the chance of the model correctly classifying all positive data against the chance of incorrectly labelling the data as positive, as well as this a area under curve calculation can be used to show how effective the model is at identifying positive data correctly    

The results from the roc curve show models effectiveness with a quick climb to the 0.7 true positive rate at where it has a 0.2 chance of falsely identifying the data as positive. This combined with a auc score of 0.8 shows the model does well to identify positive data.
#### Precision recall curve
A precision recall curve will help show how accurate the model is at correctly identifying true positives as the model increases the rate of how often it identifies a data as positives.  

Having the precision of the model stay above 0.8 until a recall of 0.7 as well as a auc score of 0.8 shows that the model performs well in correctly identifying true positives when recall is high.  


## Model 2 - SVM

To compare the results of CNN, Support vector machine (SVM) will be used.
This method was chosen due to it’s popularity in the medical field as a model that can be used as a benchmark to hold other models to its standard a seen with Tripathy (2022) using the model when comparing models for brain tumor classification.


### 5.1: Hyper parameter tuning

To optimise the model the C variable will be explored with different combinations to ensure that the best value is used. To this a Gridsearch will be used to fit the model with the hyper parameter test values and then use this data to run the svm model with each value and return the best value to set C, As there is only one value to set an Exhaustive grid search method is used to check all possible combinations.


```python
warnings.filterwarnings('ignore')
hyper_parameters = {'C':[1,10,100]}
svc_model = LinearSVC()
tuned_svm_model = GridSearchCV(svc_model,hyper_parameters,refit=True)
tuned_svm_model.fit(X_hp_svm,y_hp_svm)
print(tuned_svm_model.best_params_,'\n',tuned_svm_model.best_score_)
```

    {'C': 10} 
     0.5128647214854112


### 5.2 Build And Run


```python
svm_model = LinearSVC(C=tuned_svm_model.best_params_['C'],max_iter=2000,random_state=1)
svm_model.fit(X_train_svm,y_train_svm)
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearSVC(C=10, max_iter=2000, random_state=1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LinearSVC</label><div class="sk-toggleable__content"><pre>LinearSVC(C=10, max_iter=2000, random_state=1)</pre></div></div></div></div></div>




```python
svm_predicted = svm_model.predict(X_test_svm)
```

### 5.3 Results And Evaluation


```python
matrix = confusion_matrix(y_test_svm, svm_predicted)
sns.heatmap(matrix, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=['true', 'false'], yticklabels=['true', 'false'])
plt.xlabel('Predicted')
plt.ylabel('Actual');
recall = matrix[0][0]/(matrix[0][0]+matrix[0][1])
precision = matrix[0][0]/(matrix[0][0]+matrix[1][0])
fscore = 2*((precision*recall)/(precision+recall))
plt.title('SVM Metrics')
plt.text(0.7,2.5,'Metrics')
plt.text(0.5,2.7,'Accuracy: '+str((matrix[0][0]+matrix[1][-1])/(matrix[0][0]+matrix[0][-1]+matrix[1][0]+matrix[1][-1])))
plt.text(0.5,2.8,'Recall: '+str(recall))
plt.text(0.5,2.9,'Precision: '+str(precision))
plt.text(0.5,3.0,'fscore: '+str(fscore))
plt.show()
```


    
![png](/assets/report_assets/2025-06-18-ML_Assignment/output_64_0.png)
    


The performance of the model is outperformed by the CNN model with Recall,Precision and F1 score all being lower then the CNN’s results


```python
fig, ax = plt.subplots(figsize=(8,8),nrows=2,ncols=1)

false_pos_rate,true_pos_rate,t = roc_curve(y_test_svm,svm_predicted)

ax[0].plot(false_pos_rate,true_pos_rate)
ax[0].set_title('ROC Curve' )
ax[0].text(0,-0.35,'RoC area under curve score:'+str(roc_auc_score(y_test_svm,svm_predicted)))
ax[0].set_xlabel('false positvie rate')
ax[0].set_ylabel('true postitive rate')

plot = PrecisionRecallDisplay.from_predictions(y_test_svm,svm_predicted,ax=ax[1])
precision, recall, t= precision_recall_curve(np.array(y_test_svm),np.array(svm_predicted))
pr_auc = auc(recall,precision)
ax[1].set_title('Presicion-Recall Curve' )
ax[1].text(0,0.35,'Presicion-Recall area under curve score: '+str(pr_auc))
plt.subplots_adjust(wspace=0.1,hspace=0.5)
plt.show()
```


    
![png](/assets/report_assets/2025-06-18-ML_Assignment/output_66_0.png)
    


The area under curve ROC score also shows poor performance with the model slightly able at predicting true positives than what it would at random
The model does show better results in the precision recall curve showing a precision of 0.6 when recall is bellow 0.4 although as well as a auc score showing that the model is better than random it still well behind CNN in comparison.      


## 6: Comparision and evaluation  

When comparing both models CNN comes out the best with all metrics indicating that it outperforms SVM as the better classification model for this dataset. Although the CNN model does have a good performance there where limits discovered in the development of this project, the first being that the model itself is limited in terms of depth, although this means that the CNN model is quick to run its performance does not increase when adding additional layers, this could be due to vanishing gradient as weights and features are lost the further the model progresses into each layer. To address this more complicated models could be used such as ResNet which described by He (2015) as a framework that directly addresses the problem for deep neural networks.

As well as this model the dataset also gave some limitations, this is because the small amount of images could lead the model to overfitting on multiple runs, the start of overfitting can be seen in the CNN’s accuracy becoming more unstable the more epochs had been executed, this lead to the CNN models epochs being kept low to avoid overfitting of the data. To address this a dataset with more images could be chosen to provide a better training data for the model.

Another limitation encountered is the choice of svm model this is because the svm model did not scale well with the expanded and flipped dataset was and producing poor results when using the dataset meaning that it had to use the original dataset, this means that although we can still draw comparisons between the two models we cannot completely determine the comparison to be accurate, to address this in the future a model that scales better to larger datasets should be chosen.        
    


## 7: References

Ghaffari M., Sowmya A. and Oliver R., "Automated Brain Tumor Segmentation Using Multimodal Brain Scans: A Survey Based on Models Submitted to the BraTS 2012–2018 Challenges,”in IEEE Reviews in Biomedical Engineering, 13, pp. 156-168, 2020, Available at: https://doi.org/10.1109/RBME.2019.2946868

He, K. et al. (2015) ‘Deep Residual Learning for Image Recognition’. Available at: https://doi.org/10.48550/arxiv.1512.03385.

Tripathy, B., Mohanty, R. and Parida, S., 2022. Brain Tumour Detection Using Convolutional Neural Network-XGBoost. Acta Scientific COMPUTER SCIENCES Volume, 4(12).Available at: https://www.researchgate.net/publication/365428458_Brain_Tumour_Detection_Using_Convolutional_Neural_Network-XGBoost (Accessed on: 23/04/2024)


