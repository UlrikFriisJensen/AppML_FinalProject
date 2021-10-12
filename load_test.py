import numpy as np
import pandas as pd

from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator, img_to_array

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

path_prefix = './AppML_repo/FinalProject/train/'

batchSize = 128
n_epochs=15

#df_data = pd.read_pickle(path_prefix + 'CleanedData.pkl')

#df_data = df_data.sort_values('imgpaths', axis=0)

#data_train, data_test = train_test_split(df_data, test_size=0.25, random_state=42)

#y_train = data_train['Label']

#y_test = data_test['Label']

#header_best = ['Area (ABD)', 'Area (Filled)', 'Aspect Ratio', 'Biovolume (Cylinder)', 'Biovolume (P. Spheroid)', 'Biovolume (Sphere)', 'Circle Fit', 'Circularity', 'Circularity (Hu)', 'Compactness', 'Convex Perimeter', 'Convexity', 'Diameter (ABD)', 'Diameter (ESD)', 'Edge Gradient', 'Elongation', 'Feret Angle Max', 'Feret Angle Min', 'Fiber Curl', 'Fiber Straightness', 'Geodesic Aspect Ratio', 'Geodesic Length', 'Geodesic Thickness', 'Intensity', 'Length', 'Particles Per Chain', 'Perimeter', 'Roughness', 'Sigma Intensity', 'Sphere Complement', 'Sphere Count', 'Sphere Unknown', 'Sphere Volume', 'Sum Intensity', 'Symmetry', 'Transparency', 'Volume (ABD)', 'Volume (ESD)', 'Width']

#train_num = data_train[header_best]
#test_num = data_test[header_best]
#NumTransformer = StandardScaler()
#NumTransformer.fit(train_num)
#NumInput_train = NumTransformer.transform(train_num)
#NumInput_test = NumTransformer.transform(test_num)

#output_shape = ({'img':[256,256,1], 'num':[len(header_best)]}, [1])

#def generator_func_train():
#    for path, nums, label in zip(data_train['imgpaths'], NumInput_train, y_train.to_numpy(dtype=np.int32)):
#        image = img_to_array(load_img(path_prefix+path, color_mode='grayscale', target_size=(256,256)))* 1./255
#        image = image.reshape(256,256,1)
#        nums = nums.reshape(len(header_best))
#        label = label.reshape(1)
#        yield {'img': image, 'num': nums}, label

#TrainDataset = tf.data.Dataset.from_generator(generator_func_train, output_types=({'img':tf.float32,'num':tf.float32},tf.int32), output_shapes=output_shape).batch(batchSize)

#def generator_func_test():
#    for path, nums, label in zip(data_test['imgpaths'], NumInput_test, y_test.to_numpy(dtype=np.int32)):
#        image = img_to_array(load_img(path_prefix+path, color_mode='grayscale', target_size=(256,256))) * 1./255
#        image = image.reshape(256,256,1)
#        nums = nums.reshape(len(header_best))
#        label = label.reshape(1)
#        yield {'img': image, 'num': nums}, label

#TestDataset = tf.data.Dataset.from_generator(generator_func_test, output_types=({'img':tf.float32,'num':tf.float32},tf.int32), output_shapes=output_shape).batch(batchSize)


dataset_train = image_dataset_from_directory(
    path_prefix,
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="grayscale",
    batch_size=128,
    image_size=(256, 256),
    shuffle=True,
    seed=42,
    validation_split=0.25,
    subset="training",
    interpolation="bilinear",
    follow_links=False)

dataset_val = image_dataset_from_directory(
    path_prefix,
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="grayscale",
    batch_size=128,
    image_size=(256, 256),
    shuffle=True,
    seed=42,
    validation_split=0.25,
    subset="validation",
    interpolation="bilinear",
    follow_links=False)

#print(dataset_val.class_names)


model = tf.keras.models.load_model('./AppML_repo/FinalProject/Saved_models/Amalie_CNN_fullES')

model.evaluate(dataset_val)

def get_predictions_and_labels(model, dataset):
    labels = []
    predicts = []
    for x, y in dataset:
        y_pred = np.argmax(model.predict(x), axis=-1)
        for predict, label in zip(y_pred, y.numpy()):
            labels.append(label)
            predicts.append(predict)
    return labels, predicts

#true_label, pred = get_predictions_and_labels(model, dataset_val)

#cf = tf.math.confusion_matrix(true_label,pred, num_classes = 6).numpy()

def plot_confusion_matrix(cf,
                          group_names=None,
                          count=True,
                          percent=True,
                          cbar=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='viridis',
                          title=None):
    '''
    ---------
    Model:         Trained classifier.
    ds:            Test set.
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    count:         If True, show the raw number in the confusion matrix. Default is True.
    percent:       If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    '''
    
    

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:.2%}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    # MAKE THE HEATMAP VISUALIZATION
    categories = ['campanian', 'corylus', 'dust', 'grimsvotn', 'qrobur', 'qsuber']
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    plt.ylabel('True label')
    plt.xlabel('Predicted label' + stats_text)
    
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig('./AppML_repo/FinalProject/Confusion_matrix.pdf', dpi=500)
    plt.show()

class_names = ['campanian', 'corylus', 'dust', 'grimsvotn', 'qrobur', 'qsuber']

#plot_confusion_matrix(cf,class_names)

# true = 0
# false = 0
# for true_label_ref in [0,1,2,3,4,5]:
#     count = Counter(list(pred[true_label == true_label_ref]))
#     true += count[true_label_ref]
#     false += np.sum(list(count.values())) - count[true_label_ref]
#     print(f'Predictions for images of class {true_label_ref}:\n{count}\n')

# print(f'Accuracy = {true / (true + false):.2%}')

# m = tf.keras.metrics.Accuracy()
# # m = tf.keras.metrics.CategoricalAccuracy()
# m.update_state(pred, #)
# print()
# print(m.result().numpy())

# correct = 0
# incorrect = 0
# for x, y_val in dataset_val:
#     y_pred = np.argmax(model.predict(x), axis=-1)
#     for pred, true in zip(y_pred, y_val):
#         if pred == true:
#             correct += 1
#         else:
#             incorrect += 1

# print(f'One-by-one accuracy:  {correct / (correct + incorrect):.2%}')