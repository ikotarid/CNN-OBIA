import os, glob
import numpy as np
from sklearn.utils import resample

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

# Downsample the majority class
artificial_features = resample(artificial_features,
                            replace = False, # sample without replacement
                            n_samples = 20000,
                            random_state = 2)

artificial_labels = resample(artificial_labels,
                          replace = False, # sample without replacement
                          n_samples = 20000,
                          random_state = 2)
vegetation_features = resample(vegetation_features,
                            replace = False, # sample without replacement
                            n_samples = 20000,
                            random_state = 2)

vegetation_labels = resample(vegetation_labels,
                          replace = False, # sample without replacement
                          n_samples = 20000,
                          random_state = 2)

openspaces_features = resample(openspaces_features,
                            replace = False, # sample without replacement
                            n_samples = 20000,
                            random_state = 2)

openspaces_labels = resample(openspaces_labels,
                          replace = False, # sample without replacement
                          n_samples = 20000,
                          random_state = 2)

water_features = resample(water_features,
                            replace = False, # sample without replacement
                            n_samples = 20000,
                            random_state = 2)

water_labels = resample(water_labels,
                          replace = False, # sample without replacement
                          n_samples = 20000,
                          random_state = 2)

print('Number of records in balanced classes:')
print('artificial: %d, vegetation: %d, openspaces: %d, water: %d' % (artificial_labels.shape[0], vegetation_labels.shape[0], openspaces_labels.shape[0], water_labels.shape[0]))

# Save the arrays as .npy files
np.save('artificial_features.npy', artificial_features)
np.save('artificial_labels.npy', artificial_labels)
np.save('vegetation_features.npy', vegetation_features)
np.save('vegetation_labels.npy', vegetation_labels)
np.save('openspaces_features.npy', openspaces_features)
np.save('openspaces_labels.npy', openspaces_labels)
np.save('water_features.npy', water_features)
np.save('water_labels.npy', water_labels)
