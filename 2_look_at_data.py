import os, glob
import numpy as np

# Load arrays from .npy files
features = np.load('CNN_15by15_features.npy')
labels = np.load('CNN_15by15_labels.npy')

# Separate and balance the classes
artificial_features = features[labels==0]
artificial_labels = labels[labels==0]

vegetation_features = features[labels==1]
vegetation_labels = labels[labels==1]

openspaces_features = features[labels==2]
openspaces_labels = labels[labels==2]

water_features = features[labels==3]
water_labels = labels[labels==3]

print('Number of records in each class:')
print('artificial: %d, vegetation: %d, openspaces: %d, water: %d' % (artificial_labels.shape[0], vegetation_labels.shape[0], openspaces_labels.shape[0], water_labels.shape[0]))
