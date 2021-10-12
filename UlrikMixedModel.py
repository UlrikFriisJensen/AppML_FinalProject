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

# df_data = pd.read_pickle(path_prefix + 'CleanedData.pkl')

# df_data = df_data.sort_values('imgpaths', axis=0)

# data_train, data_test = train_test_split(df_data, test_size=0.25, random_state=42) # All of dataset
# # data_train, data_test = train_test_split(df_data, train_size=192, test_size=128, random_state=42) # Choose number of images in train and test

# batchSize = 128
# n_epochs=50

# datagen=ImageDataGenerator(rescale=1./255)
# # X_train_img = datagen.flow_from_dataframe(dataframe=data_train, directory=path_prefix, x_col="imgpaths", y_col=None, class_mode=None, color_mode='grayscale', shuffle=False, validate_filenames=False, target_size=(256,256), batch_size=batchSize)

# # X_test_img = datagen.flow_from_dataframe(dataframe=data_test, directory=path_prefix, x_col="imgpaths", y_col=None, class_mode=None, color_mode='grayscale', shuffle=False, validate_filenames=False, target_size=(256,256), batch_size=batchSize)

# y_train = data_train['Label']

# y_test = data_test['Label']

# # header_best = ['Area (ABD)', 'Area (Filled)', 'Aspect Ratio', 'Biovolume (Cylinder)', 'Biovolume (P. Spheroid)', 'Convex Perimeter', 'Diameter (ABD)', 'Diameter (ESD)', 'Geodesic Thickness', 'Intensity', 'Perimeter', 'Sum Intensity', 'Width']

header_best = ['Area (ABD)', 'Area (Filled)', 'Aspect Ratio', 'Biovolume (Cylinder)', 'Biovolume (P. Spheroid)', 'Biovolume (Sphere)', 'Circle Fit', 'Circularity', 'Circularity (Hu)', 'Compactness', 'Convex Perimeter', 'Convexity', 'Diameter (ABD)', 'Diameter (ESD)', 'Edge Gradient', 'Elongation', 'Feret Angle Max', 'Feret Angle Min', 'Fiber Curl', 'Fiber Straightness', 'Geodesic Aspect Ratio', 'Geodesic Length', 'Geodesic Thickness', 'Intensity', 'Length', 'Particles Per Chain', 'Perimeter', 'Roughness', 'Sigma Intensity', 'Sphere Complement', 'Sphere Count', 'Sphere Unknown', 'Sphere Volume', 'Sum Intensity', 'Symmetry', 'Transparency', 'Volume (ABD)', 'Volume (ESD)', 'Width']

# train_num = data_train[header_best]
# test_num = data_test[header_best]
# NumTransformer = StandardScaler()
# NumTransformer.fit(train_num)
# NumInput_train = NumTransformer.transform(train_num)
# NumInput_test = NumTransformer.transform(test_num)

# # Generate datasets
# output_shape = ({'img':[256,256,1], 'num':[len(header_best)]}, [1])

# def generator_func_train():
#     for path, nums, label in zip(data_train['imgpaths'], NumInput_train, y_train.to_numpy(dtype=np.int32)):
#         image = img_to_array(load_img(path_prefix+path, color_mode='grayscale', target_size=(256,256)))* 1./255
#         image = image.reshape(256,256,1)
#         nums = nums.reshape(len(header_best))
#         label = label.reshape(1)
#         yield {'img': image, 'num': nums}, label

# TrainDataset = tf.data.Dataset.from_generator(generator_func_train, output_types=({'img':tf.float32,'num':tf.float32},tf.int32), output_shapes=output_shape).batch(batchSize)

# def generator_func_test():
#     for path, nums, label in zip(data_test['imgpaths'], NumInput_test, y_test.to_numpy(dtype=np.int32)):
#         image = img_to_array(load_img(path_prefix+path, color_mode='grayscale', target_size=(256,256))) * 1./255
#         image = image.reshape(256,256,1)
#         nums = nums.reshape(len(header_best))
#         label = label.reshape(1)
#         yield {'img': image, 'num': nums}, label

# TestDataset = tf.data.Dataset.from_generator(generator_func_test, output_types=({'img':tf.float32,'num':tf.float32},tf.int32), output_shapes=output_shape).batch(batchSize)

#Model
inputImg = keras.Input(shape=(256, 256, 1), name='img')
inputNum = keras.Input(shape=(len(header_best)), name='num')

kernel_reg = l2(0.000013)

conv1 = layers.Conv2D(46, (5,5), padding='same', activation='relu', kernel_regularizer=kernel_reg)(inputImg)
pool1 = layers.MaxPooling2D(pool_size=2)(conv1)
conv2 = layers.Conv2D(31, (3,3), padding='same', activation='relu', kernel_regularizer=kernel_reg)(pool1)
pool2 = layers.MaxPooling2D(pool_size=2)(conv2)
conv3 = layers.Conv2D(80, (3,3), padding='same', activation='relu', kernel_regularizer=kernel_reg)(pool2)
pool3 = layers.MaxPooling2D(pool_size=2)(conv3)
conv4 = layers.Conv2D(119, (3,3), padding='same', activation='relu', kernel_regularizer=kernel_reg)(pool3)
pool4 = layers.MaxPooling2D(pool_size=2)(conv4)
dropImg1 = layers.Dropout(0.1)(pool4)
flat1 = layers.Flatten()(dropImg1)
denseImg1 = layers.Dense(128, activation='sigmoid')(flat1)
dropImg2 = layers.Dropout(0.15)(denseImg1)

denseNum1 = layers.Dense(50, activation='relu')(inputNum)
dropNum1 = layers.Dropout(0.01)(denseNum1)
denseNum2 = layers.Dense(50, activation='relu')(dropNum1)
dropNum2 = layers.Dropout(0.01)(denseNum2)
denseNum3 = layers.Dense(50, activation='relu')(dropNum2)
dropNum3 = layers.Dropout(0.01)(denseNum3)
denseNum4 = layers.Dense(50, activation='relu')(dropNum3)

conc1 = layers.Concatenate()([dropImg2, denseNum4])

denseComb1 = layers.Dense(128, activation='relu')(conc1)
dropComb1 = layers.Dropout(0.1)(denseComb1)
denseComb2 = layers.Dense(50, activation='relu')(dropComb1)
output = layers.Dense(6, activation='softmax', name='Output')(denseComb2)

model = keras.Model(inputs=[inputImg, inputNum], outputs=output, name='MixedModel')

model.compile(optimizer='adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics='accuracy')
model.summary()

#plot_model(model, to_file='./AppML_repo/FinalProject/UlrikMixedModel_ES2_architechture_png.png', rankdir='TB', show_shapes=True, dpi=500)

# es = EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True) # min_delta=5e-3

# history = model.fit(TrainDataset,validation_data=TestDataset, epochs=n_epochs, verbose=1, callbacks=[es])

# model.save('./AppML_repo/FinalProject/Saved_models/UlrikMixedModel_ES2')

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# def plot_loss(losses,accs):
#   epochs_range = range(len(losses[0]))

#   sns.set()

#   plt.figure(figsize=(12, 6))
#   plt.subplot(1, 2, 1)
#   plt.plot(epochs_range, accs[0], label='Training Accuracy')
#   plt.plot(epochs_range, accs[1], label='Validation Accuracy')
#   plt.legend(loc='lower right')
#   plt.xlabel('Epochs')
#   plt.ylabel('Accuracy')
#   plt.title('Training and Validation Accuracy')

#   plt.subplot(1, 2, 2)
#   plt.plot(epochs_range, losses[0], label='Training Loss')
#   plt.plot(epochs_range, losses[1], label='Validation Loss')
#   plt.legend(loc='upper right')
#   plt.xlabel('Epochs')
#   plt.ylabel('Loss')
#   plt.title('Training and Validation Loss')
#   plt.tight_layout()
#   plt.savefig('./AppML_repo/FinalProject/UlrikMixedModel_ES2_loss.pdf', format='pdf')
# #   plt.show()

# plot_loss([loss, val_loss], [acc, val_acc])

# def get_predictions_and_labels(model, dataset):
#     labels = []
#     predicts = []
#     for x, y in dataset:
#         y_pred = np.argmax(model.predict(x), axis=-1)
#         for predict, label in zip(y_pred, y.numpy()):
#             labels.append(label)
#             predicts.append(predict)
#     return labels, predicts

# true_label, y_pred = get_predictions_and_labels(model, TestDataset)

# cf = tf.math.confusion_matrix(true_label,y_pred, num_classes = 6).numpy()

# def plot_confusion_matrix(cf, filename,
#                           group_names=None,
#                           count=True,
#                           percent=True,
#                           cbar=True,
#                           sum_stats=True,
#                           figsize=None,
#                           cmap='viridis',
#                           title=None):
#     '''
#     ---------
#     Model:         Trained classifier.
#     ds:            Test set.
#     group_names:   List of strings that represent the labels row by row to be shown in each square.
#     count:         If True, show the raw number in the confusion matrix. Default is True.
#     percent:       If True, show the proportions for each category. Default is True.
#     cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
#                    Default is True.
#     sum_stats:     If True, display summary statistics below the figure. Default is True.
#     figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
#     cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
#                    See http://matplotlib.org/examples/color/colormaps_reference.html
                   
#     title:         Title for the heatmap. Default is None.
#     '''
    
    

#     # CODE TO GENERATE TEXT INSIDE EACH SQUARE
#     blanks = ['' for i in range(cf.size)]

#     if group_names and len(group_names)==cf.size:
#         group_labels = ["{}\n".format(value) for value in group_names]
#     else:
#         group_labels = blanks

#     if count:
#         group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
#     else:
#         group_counts = blanks

#     if percent:
#         group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
#     else:
#         group_percentages = blanks

#     box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
#     box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


#     # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
#     if sum_stats:
#         #Accuracy is sum of diagonal divided by total observations
#         accuracy  = np.trace(cf) / float(np.sum(cf))

#         #if it is a binary confusion matrix, show some more stats
#         if len(cf)==2:
#             #Metrics for Binary Confusion Matrices
#             precision = cf[1,1] / sum(cf[:,1])
#             recall    = cf[1,1] / sum(cf[1,:])
#             f1_score  = 2*precision*recall / (precision + recall)
#             stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
#                 accuracy,precision,recall,f1_score)
#         else:
#             stats_text = "\n\nAccuracy={:.2%}".format(accuracy)
#     else:
#         stats_text = ""


#     # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
#     if figsize==None:
#         #Get default figure size if not set
#         figsize = plt.rcParams.get('figure.figsize')

#     # MAKE THE HEATMAP VISUALIZATION
#     categories = ['campanian', 'corylus', 'dust', 'grimsvotn', 'qrobur', 'qsuber']
#     plt.figure(figsize=figsize)
#     sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

#     plt.ylabel('True label')
#     plt.xlabel('Predicted label' + stats_text)
    
#     if title:
#         plt.title(title)
#     plt.tight_layout()
#     plt.savefig(f'./AppML_repo/FinalProject/ConfusionMatrix_{filename}.pdf', dpi=500)
#     # plt.show()

# class_names = ['campanian', 'corylus', 'dust', 'grimsvotn', 'qrobur', 'qsuber']

# plot_confusion_matrix(cf,'UlrikMixedModel_ES2',group_names=class_names, title='Mixed Model')