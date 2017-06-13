import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import LabelEncoder

class ModelGenerator:
    def generate_text_rec(self, captcha_size, alphabet_size, sample_size, key_mode, model_name):
        import numpy as np
        keys_array = np.array([None] * sample_size)

        import subprocess
        import cv2

        sample_array = None
        is_array_generated = False
        (img_rows, img_cols) = (0, 0)
        for i in range(sample_size):
            s = subprocess.check_output(['php', 'kcaptcha/index.php'])
            img = cv2.imdecode(np.frombuffer(s, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if not is_array_generated:
                (img_rows, img_cols) = img.shape
                sample_array = np.empty((sample_size,) + img.shape)
                is_array_generated = True
            sample_array[i] = img

            key_filename = './key'
            file_mode = 'r'

            if key_mode == 'str':
                keys_array[i] = open(key_filename, file_mode).read()
            elif key_mode == 'len':
                keys_array[i] = len(open(key_filename, file_mode).read())

        training_size = int(sample_size * 0.8)
        training_array = sample_array[:training_size]
        test_array = sample_array[training_size:]

        training_array = training_array.reshape(training_array.shape[0], img_rows, img_cols, 1)
        test_array = test_array.reshape(test_array.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

        encoder = LabelEncoder()
        encoder.fit(keys_array)
        encoded_keys_array = encoder.transform(keys_array)

        print encoded_keys_array.max()

        keys_categorial_array = keras.utils.to_categorical(encoded_keys_array, num_classes=alphabet_size * captcha_size)
        keys_training_array = keys_categorial_array[:training_size]
        keys_test_array = keys_categorial_array[training_size:]

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

        model.add(Dense(alphabet_size * captcha_size, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        model.fit(training_array, keys_training_array,
                  batch_size=64,
                  epochs=12,
                  verbose=1,
                  validation_data=(test_array, keys_test_array))
        print model.evaluate(test_array, keys_test_array, verbose=0)

        model.save('models/' + model_name + '.h5')
        model.save_weights('weights/' + model_name + '_weights.h5')