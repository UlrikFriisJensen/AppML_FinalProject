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
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator, img_to_array
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

path_prefix = './AppML_repo/FinalProject/'

icecore_path = 'test_GRIP_31may2021/GRIP_raw/'

icecore_id = 'GRIP_3306'

batchSize = 128
n_epochs=50

df_data = pd.read_pickle(path_prefix + 'CleanedData.pkl')

df_icecore = pd.read_csv(path_prefix + icecore_path + icecore_id + '_raw.csv')
df_icecore['imgpaths'] = df_icecore['imgpaths'].str.split('GRIP_raw/').str[1]

df_data = df_data.sort_values('imgpaths', axis=0)
df_icecore = df_icecore.sort_values('imgpaths', axis=0)
print(len(df_icecore['imgpaths']))

header_best = ['Area (ABD)', 'Area (Filled)', 'Aspect Ratio', 'Biovolume (Cylinder)', 'Biovolume (P. Spheroid)', 'Biovolume (Sphere)', 'Circle Fit', 'Circularity', 'Circularity (Hu)', 'Compactness', 'Convex Perimeter', 'Convexity', 'Diameter (ABD)', 'Diameter (ESD)', 'Edge Gradient', 'Elongation', 'Feret Angle Max', 'Feret Angle Min', 'Fiber Curl', 'Fiber Straightness', 'Geodesic Aspect Ratio', 'Geodesic Length', 'Geodesic Thickness', 'Intensity', 'Length', 'Particles Per Chain', 'Perimeter', 'Roughness', 'Sigma Intensity', 'Sphere Complement', 'Sphere Count', 'Sphere Unknown', 'Sphere Volume', 'Sum Intensity', 'Symmetry', 'Transparency', 'Volume (ABD)', 'Volume (ESD)', 'Width']


data_train, data_test = train_test_split(df_data, test_size=0.25, random_state=42)

train_num = data_train[header_best]
test_num = data_test[header_best]
icecore_num = df_icecore[header_best]
NumTransformer = StandardScaler()
NumTransformer.fit(train_num)
NumInput_train = NumTransformer.transform(train_num)
NumInput_test = NumTransformer.transform(test_num)
NumInput_icecore = NumTransformer.transform(icecore_num)

y_train = data_train['Label']

y_test = data_test['Label']

output_shape = ({'img':[256,256,1], 'num':[len(header_best)]}, [1])

def generator_func_train():
    for path, nums, label in zip(data_train['imgpaths'], NumInput_train, y_train.to_numpy(dtype=np.int32)):
        image = img_to_array(load_img(path_prefix+path, color_mode='grayscale', target_size=(256,256)))* 1./255
        image = image.reshape(256,256,1)
        nums = nums.reshape(len(header_best))
        label = label.reshape(1)
        yield {'img': image, 'num': nums}, label

TrainDataset = tf.data.Dataset.from_generator(generator_func_train, output_types=({'img':tf.float32,'num':tf.float32},tf.int32), output_shapes=output_shape).batch(batchSize)

def generator_func_test():
    for path, nums, label in zip(data_test['imgpaths'], NumInput_test, y_test.to_numpy(dtype=np.int32)):
        image = img_to_array(load_img(path_prefix+path, color_mode='grayscale', target_size=(256,256))) * 1./255
        image = image.reshape(256,256,1)
        nums = nums.reshape(len(header_best))
        label = label.reshape(1)
        yield {'img': image, 'num': nums}, label

TestDataset = tf.data.Dataset.from_generator(generator_func_test, output_types=({'img':tf.float32,'num':tf.float32},tf.int32), output_shapes=output_shape).batch(batchSize)

def generator_func_icecore():
    for path, nums in zip(df_icecore['imgpaths'], NumInput_icecore):
        image = img_to_array(load_img(path_prefix+icecore_path+path, color_mode='grayscale', target_size=(256,256))) * 1./255
        image = image.reshape(256,256,1)
        nums = nums.reshape(len(header_best))
        image_path = np.array([icecore_path+path])
        yield {'img': image, 'num': nums}, image_path

IcecoreDataset = tf.data.Dataset.from_generator(generator_func_icecore, output_types=({'img':tf.float32,'num':tf.float32}, tf.string), output_shapes=output_shape).batch(batchSize)

model = tf.keras.models.load_model('./AppML_repo/FinalProject/Saved_models/UlrikMixedModel_ES2')

class_names = ['campanian', 'corylus', 'dust', 'grimsvotn', 'qrobur', 'qsuber']

def get_predictions_and_path(model, dataset):
    paths = []
    predicts = []
    for x, y in dataset:
        y_pred = np.argmax(model.predict(x, verbose=0), axis=-1)
        for predict, path in zip(y_pred, y.numpy()):
            paths.append(path)
            predicts.append(predict)
    return paths, predicts

paths, predicts = get_predictions_and_path(model, IcecoreDataset)

with open(f'./AppML_repo/FinalProject/IcecoreData_Predictions_{icecore_id}.txt', 'w') as f:
    for path, pred in zip(paths, predicts):
        f.write(f'{path[0]}, {pred}')
        f.write('\n')