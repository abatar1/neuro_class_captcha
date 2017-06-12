import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from keras.layers.normalization import BatchNormalization

import numpy as np

class ModelGenerator:
    def generate_text(self, generator_path, sample_size, key_mode):
        keys_array = np.array([None] * sample_size)

        import subprocess
        import cv2

        sample_array = None
        is_array_generated = False
        img_shape = 0
        for i in range(sample_size):
            try:
                s = subprocess.check_output(['php', generator_path])
            except subprocess.CalledProcessError:
                print 'Called process error!'

            img = cv2.imdecode(np.frombuffer(s, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

            if not is_array_generated:
                img_shape = img.shape
                sample_array = np.empty((sample_size,) + img.shape)
                is_array_generated = True
            sample_array[i] = img

            key_filename = './key'
            file_mode = 'r'

            if key_mode == 'str':
                keys_array[i] = open(key_filename, file_mode).read()
            elif key_mode == 'len':
                keys_array[i] = len(open(key_filename, file_mode).read())
        return (sample_array, keys_array, img_shape)

    def categorize_keys(self, keys_array, alphabet, num_classes):
        from sklearn.preprocessing import LabelBinarizer

        alphabet_list = list(alphabet)
        alphabet_len = len(alphabet)

        binarizer = LabelBinarizer()
        binarizer.fit(keys_array)

        binarized_keys_array = np.array([None] * alphabet_len)
        i = 0
        for c in alphabet_list:
            binarized_keys_array[i] = binarizer.transform(c)
            i += 1

        return np.concatenate(binarized_keys_array)

    def build_model(self, input_shape, classes_num):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv2D(64, kernel_size=(3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(classes_num, activation='softmax'))

        return model

    def generate(self, generator_path, sample_size, key_mode, alphabet):
        (sample_array, keys_array, img_shape) = self.generate_text(generator_path=generator_path,
                           sample_size=sample_size,
                           key_mode=key_mode)

        captcha_size = len(keys_array[0])

        training_size = int(sample_size * 0.8)
        training_array = sample_array[:training_size]
        test_array = sample_array[training_size:]

        (img_rows, img_cols) = img_shape

        from keras import backend as K

        if K.image_data_format() == 'channels_first':
            training_array = training_array.reshape(training_array.shape[0], 1, img_rows, img_cols)
            test_array = test_array.reshape(test_array.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            training_array = training_array.reshape(training_array.shape[0], img_rows, img_cols, 1)
            test_array = test_array.reshape(test_array.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

        num_classes = len(alphabet) * captcha_size
        keys_categorial = self.categorize_keys(keys_array=keys_array, alphabet=alphabet, num_classes=num_classes)

        keys_training = keys_categorial[:training_size]
        keys_test = keys_categorial[training_size:]

        model = self.build_model(input_shape=input_shape, )

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        model.fit(training_array, keys_training,
                  batch_size=64,
                  epochs=12,
                  verbose=1,
                  validation_data=(test_array, keys_test))

        model_result = model.evaluate(test_array, keys_test, verbose=0)
        return (model, model_result)