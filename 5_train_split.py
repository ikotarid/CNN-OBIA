# Define the function to split features and labels
import numpy as np
import random
import tensorflow as tf

features = np.load('features_norm.npy')
labels = np.load('labels_norm.npy')

def train_test_split(features, labels, trainProp=0.7):
    dataSize = features.shape[0]
    sliceIndex = int(dataSize*trainProp)
    randIndex = np.arange(dataSize)
    random.shuffle(randIndex)
    train_x = features[[randIndex[:sliceIndex]], :, :, :][0]
    test_x = features[[randIndex[sliceIndex:]], :, :, :][0]
    train_y = labels[randIndex[:sliceIndex]]
    test_y = labels[randIndex[sliceIndex:]]
    return(train_x, train_y, test_x, test_y)

# Call the function to split the data
train_x, train_y, test_x, test_y = train_test_split(features, labels)

print('Reshaped split features:', train_x.shape, test_x.shape)
print('Split labels:', train_y.shape, test_y.shape)

np.save('train_x.npy', train_x)
np.save('test_x.npy', test_x)
np.save('train_y.npy', train_y)
np.save('test_y.npy', test_y)
