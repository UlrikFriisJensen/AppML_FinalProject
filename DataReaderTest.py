import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import load_img

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

path_prefix = './AppML_repo/FinalProject/'

df_data = pd.read_csv(path_prefix + 'CleanedData.csv')

df_data = df_data.sort_values('imgpaths', axis=0)

dataset_train = image_dataset_from_directory(
    path_prefix + 'train/',
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="grayscale",
    batch_size=128,
    image_size=(256, 256),
    shuffle=False,
    seed=42,
    validation_split=0.25,
    subset="training",
    interpolation="bilinear",
    follow_links=False)

# dataset_val = image_dataset_from_directory(
#     path_prefix,
#     labels="inferred",
#     label_mode="int",
#     class_names=None,
#     color_mode="grayscale",
#     batch_size=128,
#     image_size=(256, 256),
#     shuffle=False,
#     seed=42,
#     validation_split=0.25,
#     subset="validation",
#     interpolation="bilinear",
#     follow_links=False)

for image_, label_ in dataset_train.take(1):
    print(label_)
#   plt.imshow(image_[0,:,:,0])
    # print(dataset_train.class_names[label_[0]])
#   plt.show()

# plt.imshow(load_img(path_prefix + df_data['imgpaths'][0]))
# plt.show()