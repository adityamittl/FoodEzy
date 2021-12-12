# import the required libraries

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Global Variables for later use

IMAGE_SIZE = 128
BATCH_SIZE = 32
CHANNELS = 3

# There are 4000 Images in our Dataset belonging to 80 different Classes

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "../input/indian-food-images-dataset",
    seed=123, # this will ensure we get the same images each time
    shuffle=True, # images inside the batches will be shuffled
    image_size=(IMAGE_SIZE,IMAGE_SIZE), # every image will be of 256x256 dimention
    batch_size=BATCH_SIZE # There will be 32 images in each batch
)

# The Images belong to the following Class Labels
'''
Class names are as follows-
['adhirasam',
 'aloo_gobi',
 'aloo_matar',
 'aloo_methi',
 'aloo_shimla_mirch',
 'aloo_tikki',
 'anarsa',
 'ariselu',
 'bandar_laddu',
 'basundi',
 'bhatura',
 'bhindi_masala',
 'biryani',
 'boondi',
 'butter_chicken',
 'chak_hao_kheer',
 'cham_cham',
 'chana_masala',
 'chapati',
 'chhena_kheeri',
 'chicken_razala',
 'chicken_tikka',
 'chicken_tikka_masala',
 'chikki',
 'daal_baati_churma',
 'daal_puri',
 'dal_makhani',
 'dal_tadka',
 'dharwad_pedha',
 'doodhpak',
 'double_ka_meetha',
 'dum_aloo',
 'gajar_ka_halwa',
 'gavvalu',
 'ghevar',
 'gulab_jamun',
 'imarti',
 'jalebi',
 'kachori',
 'kadai_paneer',
 'kadhi_pakoda',
 'kajjikaya',
 'kakinada_khaja',
 'kalakand',
 'karela_bharta',
 'kofta',
 'kuzhi_paniyaram',
 'lassi',
 'ledikeni',
 'litti_chokha',
 'lyangcha',
 'maach_jhol',
 'makki_di_roti_sarson_da_saag',
 'malapua',
 'misi_roti',
 'misti_doi',
 'modak',
 'mysore_pak',
 'naan',
 'navrattan_korma',
 'palak_paneer',
 'paneer_butter_masala',
 'phirni',
 'pithe',
 'poha',
 'poornalu',
 'pootharekulu',
 'qubani_ka_meetha',
 'rabri',
 'ras_malai',
 'rasgulla',
 'sandesh',
 'shankarpali',
 'sheer_korma',
 'sheera',
 'shrikhand',
 'sohan_halwa',
 'sohan_papdi',
 'sutar_feni',
 'unni_appam']
'''
class_names = dataset.class_names

'''
Splitting the dataset
We will take -
- 80% data for training our model
- 10% data for validation purpose
- 10% data for test purpose
'''

# total batches of data = 125

train_ds = dataset.take(100) # 80% of 125
test_ds = dataset.skip(100) # remaining 20%
val_ds = test_ds.take(12) # 10% of 125
test_ds = test_ds.skip(12) # 10% of 125

# Cache, Shuffle and Prefetch the dataset to train the model faster

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

# Data Augmentation Layer 
# Augmentation is the process of creating new training samples by altering the available data.

data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2),
])

# Applying Augmentation on Training Data
train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y)
).prefetch(buffer_size=tf.data.AUTOTUNE)

# Designing and Training the Model

# Reshaping so that each image is of same size and rescaling images them for normalization
resize_and_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.experimental.preprocessing.Rescaling(1./255),
])

input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 80

# creating a sequential model
model = tf.keras.Sequential([
    resize_and_rescale,
    Conv2D(filters=32, kernel_size=(2,2), activation='relu', input_shape=input_shape),
    MaxPooling2D((4,4)),
    
    Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'),
    MaxPooling2D((3,3)),
    Dropout(0.3), # for regularization
    
    Conv2D(filters=64, kernel_size=(4,4), activation='relu', padding='same'),
    Conv2D(filters=128, kernel_size=(5,5), activation='relu', padding='same'),
    MaxPooling2D((2,2)),
    Dropout(0.4),
    
    Conv2D(filters=128, kernel_size=(5,5), activation='relu', padding='same'),
    MaxPooling2D((2,2)),
    Dropout(0.5),
    
    Flatten(), # flattening for feeding into ANN
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dense(n_classes, activation='softmax')
])

model.build(input_shape=input_shape)

# Slowing down the learning rate
opt = optimizers.Adam(learning_rate=0.0001)

# compile the model
model.compile(loss = 'sparse_categorical_crossentropy', optimizer=opt, metrics= ["accuracy"])

# use early stopping to exit training if validation loss is not decreasing even after certain epochs (patience)
earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

# save the best model with least validation loss
checkpointer = ModelCheckpoint(filepath="foodclassifier_weights.h5", verbose=1, save_best_only=True)

history = model.fit(train_ds, epochs = 1000, validation_data=val_ds, batch_size=BATCH_SIZE, shuffle=True, callbacks=[earlystopping, checkpointer])

# save the model architecture to json file for future use

model_json = model.to_json()
with open("foodclassifier_model.json","w") as json_file:
    json_file.write(model_json)

# Load pretrained model (best saved one)
with open('foodclassifier_model.json', 'r') as json_file:
    json_savedModel= json_file.read()
# load the model  
model = tf.keras.models.model_from_json(json_savedModel)
model.load_weights('foodclassifier_weights.h5')
model.compile(loss = 'sparse_categorical_crossentropy', optimizer=opt, metrics= ["accuracy"])

scores = model.evaluate(test_ds)

print(scores)