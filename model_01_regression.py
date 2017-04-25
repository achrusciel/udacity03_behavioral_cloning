# region Import external libraries
import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
# endregion

# region Standard parameters, e.g. file paths
with open('./configuration/config.json', 'r') as f:
    config = json.load(f)
local_path = config['local_path']
# endregion

# region Collecting data from file system
data = []
with open(local_path + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        data.append(line)

train_samples, validation_samples = train_test_split(data, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                source_path = line[0]
                filename = source_path.split('/')[-1]
                current_path = local_path + 'IMG/' + filename
                center_image = cv2.imread(current_path)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# endregion

# region Hyperparameters
EPOCHS = 10
# endregion

# region the model itself
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = Sequential()
model.add(Cropping2D(cropping=((60, 25), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Flatten())
model.add(Dense(1))

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
model.save('model_01.h5')
# endregion
