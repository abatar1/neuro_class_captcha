alphabet_size = 10
letter_sample_size = 20
sample_size = alphabet_size * letter_sample_size

import numpy as np
keys_array = np.array([None] * sample_size)

import subprocess
import cv2

sample_array = None
is_array_generated = False
(img_rows, img_cols) = (0,0)
for i in range(sample_size):
    s = subprocess.check_output(['php', 'kcaptcha/index.php'])
    img = cv2.imdecode(np.frombuffer(s, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if not is_array_generated:
        (img_rows, img_cols) = img.shape
        sample_array = np.empty((sample_size,) + img.shape)
        is_array_generated = True
    sample_array[i] = img
    keys_array[i] = int(open('./key', 'r').read())

training_size = int(sample_size * 0.8)
training_array = sample_array[:training_size]
test_array = sample_array[training_size:]

training_array = training_array.reshape(training_array.shape[0], img_rows, img_cols, 1)
test_array = test_array.reshape(test_array.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

keys_categorial_array = keras.utils.to_categorical(keys_array, num_classes=alphabet_size)
keys_training_array = keys_categorial_array[:training_size]
keys_test_array = keys_categorial_array[training_size:]

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(alphabet_size, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(training_array, keys_training_array,
          batch_size=128,
          epochs=12,
          verbose=1,
          validation_data=(test_array, keys_test_array))
print model.evaluate(test_array, keys_test_array, verbose=0)

model.save('1_model.h5')
model.save_weights('1_model_weights.h5')