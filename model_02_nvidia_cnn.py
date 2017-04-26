# region Import external libraries
import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout, ELU
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import json
# endregion

# region Standard parameters, e.g. file paths
with open('./configuration/config.json', 'r') as f:
    config = json.load(f)

DATA_BUCKET = 'data05_agg'
TRAINING_DATA_PATH = config['local_path'] + DATA_BUCKET + '/'
MODEL_NAME = 'model_02.h5'
BATCH_SIZE_DATA_GENERATOR = 64
# endregion

# region Collecting data from file system
data = []

with open(TRAINING_DATA_PATH + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        data.append(line)

train_samples, validation_samples = train_test_split(data, test_size=0.2)


def generator(samples, batch_size):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                source_path = batch_sample[0]
                filename = source_path.split('/')[-1]
                current_path = TRAINING_DATA_PATH + 'IMG/' + filename
                center_image = cv2.imread(current_path)
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                # Data augmentation
                flipped_image = center_image.copy()
                flipped_image = cv2.flip(flipped_image, 1)
                flipped_angle = center_angle * (-1.)
                # Store training data
                images.append(center_image)
                images.append(flipped_image)
                angles.append(center_angle)
                angles.append(flipped_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# endregion

# region Hyperparameters
EPOCHS = 5
# endregion

# region the model itself
train_generator = generator(train_samples, batch_size=BATCH_SIZE_DATA_GENERATOR)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE_DATA_GENERATOR)

# This model resembles the NVIDIA CNN described in [Bojarski et al. 2016]

model = Sequential()
# Crop off disturbing information; everything above the horizon. The values are derived manually.
model.add(Cropping2D(cropping=((60, 25), (0, 0)), input_shape=(160, 320, 3)))
# normalization
model.add(Lambda(lambda x: x / 255.0 - 0.5))
# Several convolutional layers, each followed by ELU activation
# 8x8 convolution (kernel) with 4x4 stride over 16 output filters
model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
model.add(ELU())
# 5x5 convolution (kernel) with 2x2 stride over 32 output filters
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(ELU())
# 5x5 convolution (kernel) with 2x2 stride over 64 output filters
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
# Flatten the input to the next layer
model.add(Flatten())
# Apply dropout to reduce overfitting
model.add(Dropout(.2))
model.add(ELU())
# Fully connected layer
model.add(Dense(512))
# More dropout
model.add(Dropout(.5))
model.add(ELU())
# Fully connected layer with one output dimension (representing the speed).
model.add(Dense(1))
# Compile the model
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
                                     validation_data=validation_generator,
                                     nb_val_samples=len(validation_samples), nb_epoch=EPOCHS)
# endregion

# region plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
# endregion

# region Save the model
print('Saving model with name ' + MODEL_NAME)
model.save(MODEL_NAME)
print('Model saving complete')
# endregion
