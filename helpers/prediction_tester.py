import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from keras.models import load_model
import h5py
from keras import __version__ as keras_version

model = load_model('../model_02.h5')
image = Image.open('../training_data/data03_test/IMG/center_2016_12_01_13_30_48_404.jpg')
image_array = np.asarray(image)
steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))
print(steering_angle)
