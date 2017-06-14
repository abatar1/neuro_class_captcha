import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from keras.layers.normalization import BatchNormalization

import numpy as np


class ModelGenerator:
    def __init__(self):
        pass

    @staticmethod
    def generate_text(generator_path, sample_size, key_mode):
        keys_array = np.array([None] * sample_size)

        import subprocess
        import cv2

        sample_array = None
        is_array_generated = False
        img_shape = 0

        for i in range(sample_size):
            s = subprocess.check_output(['php', generator_path])
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

        return sample_array, keys_array, img_shape

    @staticmethod
    def categorize_keys(keys_array, alphabet, key_len):
        from sklearn.preprocessing import LabelBinarizer

        binarizer = LabelBinarizer()
        binarizer.fit(list(alphabet))
        
        binarized_keys_array = np.empty([len(keys_array), len(alphabet) * key_len])
        i = 0
        for c in keys_array:
            ct = binarizer.transform(list(c) if key_len > 1 else [c])
            binarized_keys_array[i] = np.concatenate(ct)
            i += 1

        return binarized_keys_array

    @staticmethod
    def build_model(input_shape, num_classes):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(64, kernel_size=(3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, padding='same', kernel_size=(3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(64, kernel_size=(3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        return model
    
    def generate(self, generator_path, sample_size, key_mode, alphabet):
        (sample_array, keys_array, img_shape) = self.generate_text(generator_path=generator_path,
                                                                   sample_size=sample_size,
                                                                   key_mode=key_mode)
        training_size = int(sample_size * 0.8)
        training_array = sample_array[:training_size]
        test_array = sample_array[training_size:]

        (img_rows, img_cols) = img_shape

        from keras import backend as k

        if k.image_data_format() == 'channels_first':
            training_array = training_array.reshape(training_array.shape[0], 1, img_rows, img_cols)
            test_array = test_array.reshape(test_array.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            training_array = training_array.reshape(training_array.shape[0], img_rows, img_cols, 1)
            test_array = test_array.reshape(test_array.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

        if key_mode == 'len':
            classified_symbols_num = 1
        elif key_mode == 'str':
            classified_symbols_num = len(keys_array[0])
        else:
            return None

        keys_categorial = self.categorize_keys(keys_array=keys_array, alphabet=alphabet, key_len=classified_symbols_num)
        keys_training = keys_categorial[:training_size]
        keys_test = keys_categorial[training_size:]

        num_classes = len(alphabet) * classified_symbols_num
        model = self.build_model(input_shape=input_shape, num_classes=num_classes)

        model.fit(training_array, keys_training,
                  batch_size=64,
                  epochs=12,
                  verbose=1,
                  validation_data=(test_array, keys_test))

        model_result = model.evaluate(test_array, keys_test, verbose=0)
        return model, model_result
