import os, glob
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
from datetime import datetime

train_x = np.load('train_x.npy')
test_x = np.load('test_x.npy')
train_y = np.load('train_y.npy')
test_y = np.load('test_y.npy')

# Define a Keras model
model = keras.Sequential()
model.add(Conv2D(32, kernel_size=3, padding='valid', activation='relu', input_shape=(15, 15, 7)))
model.add(MaxPooling2D(2, 2))
#model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=3, padding='valid', activation='relu'))
model.add(MaxPooling2D(2, 2))
#model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(4, activation='softmax'))

print(model.summary()) #added 9/10

model.compile(loss='sparse_categorical_crossentropy', optimizer= 'adam', metrics=['accuracy']) #optimizers (adam, rmsprop)

# Define the Keras TensorBoard callback.
logdir=r"C:\Users\laptop\Desktop\cnn\_thessaloniki_15_15_object_indices\logs\fit_"+ datetime.now().strftime("%Y%m%d_%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

# cmd: tensorboard --logdir logs
# browser: http://localhost:6006/

# Train the model
model.fit(train_x, train_y, epochs=10, validation_data=(test_x, test_y), callbacks=[tensorboard_callback]) #epochs=? batch_size=?

#Evaluate the model
#test_loss, test_acc = model.evaluate(test_x, test_y) #added 9/10
#print('\nTest accuracy:', test_acc)

# Predict for test data
y_pred = model.predict(test_x)
#y_pred = y_pred.argmax(axis=-1)
y_classes = [np.argmax(element) for element in y_pred]

# Calculate and display the error metrics
cMatrix = confusion_matrix(test_y, y_classes)
class_report = classification_report(test_y, y_classes) #added 11/9
print("Confusion matrix:\n", cMatrix)
print(class_report)#added 11/9

# Save the model inside a folder to use later
if not os.path.exists(os.path.join(os.getcwd(), 'trained_models')):
    os.mkdir(os.path.join(os.getcwd(), 'trained_models'))

model.save('trained_models/CNN_nulticlass_object_indices_demo4del.h5')
