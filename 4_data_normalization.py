# Combine the balanced features
import numpy as np

artificial_features = np.load('artificial_features.npy')
artificial_labels = np.load('artificial_labels.npy')
vegetation_features = np.load('vegetation_features.npy')
vegetation_labels = np.load('vegetation_labels.npy')
openspaces_features = np.load('openspaces_features.npy')
openspaces_labels = np.load('openspaces_labels.npy')
water_features = np.load('water_features.npy')
water_labels = np.load('water_labels.npy')

features = np.concatenate((artificial_features, vegetation_features, openspaces_features, water_features), axis=0)
labels = np.concatenate((artificial_labels, vegetation_labels, openspaces_labels, water_labels), axis=0)

print('Values in input features, min: %.2f & max: %.2f' % (features.min(), features.max()))

np.save('features_norm.npy', features)
np.save('labels_norm.npy', labels)
