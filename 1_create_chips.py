import numpy as np
from pyrsgis import raster
from pyrsgis.ml import imageChipsFromFile

# define the file names
feature_file = r"C:\Users\laptop\Desktop\cnn\data\thessaloniki_1_8_2021\indices\east_thess_felzen.tif"
label_file = r"C:\Users\laptop\Desktop\cnn\data\thessaloniki_1_8_2021\indices\east_thess_felzen_class.tif"

# create feature chips using pyrsgis
features = imageChipsFromFile(feature_file, x_size=15, y_size=15)

#features = np.rollaxis(features, 3, 1)

# read the label file and reshape it
ds, labels = raster.read(label_file)
labels = labels.flatten()

# check for irrelevant values (we are interested in 1s and non-1s)
labels = labels.astype(int)

# Save the arrays as .npy files
np.save('CNN_15by15_features.npy', features)
np.save('CNN_15by15_labels.npy', labels)

# print basic details
print('Input features shape:', features.shape)
print('\nInput labels shape:', labels.shape)
print('Values in input features, min: %d & max: %d' % (features.min(), features.max()))
