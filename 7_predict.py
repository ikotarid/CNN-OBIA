import tensorflow as tf
import numpy as np
from pyrsgis import raster
from pyrsgis.ml import raster_to_chips

# Load the saved model
model = tf.keras.models.load_model(r'C:\Users\laptop\Desktop\cnn\_thessaloniki_15_15_object_indices\trained_models\CNN_nulticlass_object_indices.h5')

# Load a new multispectral image
ds, features_multiclass_ = raster.read(r'C:\Users\laptop\Desktop\cnn\data\patra_1_8_2021\indices\patra_felzen.tif')
features_multiclass = r'C:\Users\laptop\Desktop\cnn\data\patra_1_8_2021\indices\patra_felzen.tif'


# Generate image chips from array
""" Update: 29 May 2021
Note that this time we are generating image chips from array
because we needed the datasource object (ds) to export the TIF file.
And since we are reading the TIF file anyway, why not use the array directly.
"""
new_features = raster_to_chips(features_multiclass, x_size=15, y_size=15)

print('Shape of the new features', new_features.shape)

# Predict new data and export the probability raster
newPredicted = model.predict(new_features)
print("Shape of the predicted labels: ", newPredicted.shape)
newPredicted_classes = [np.argmax(element) for element in newPredicted]

prediction = np.reshape(newPredicted_classes, (ds.RasterYSize, ds.RasterXSize))

outFile = 'patra_multiclass_predicted_15by15_10_ep_filter_size3_norm_object_indices_final----------.tif'
raster.export(prediction, ds, filename=outFile, dtype='float')
