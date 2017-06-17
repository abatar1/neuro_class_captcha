import numpy as np
from keras import backend as k


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
        from keras.models import Sequential
        from keras.layers.core import Dense, Dropout, Flatten, Activation
        from keras.layers.convolutional import Conv2D
        from keras.layers.pooling import MaxPooling2D
        from keras.layers.normalization import BatchNormalization

        model = Sequential()
        model.add(Conv2D(16, kernel_size=(3, 3), padding='same', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Conv2D(16, padding='same', kernel_size=(3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(16, padding='same', kernel_size=(3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(16, padding='same', kernel_size=(3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(16, padding='same', kernel_size=(3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(16, padding='same', kernel_size=(3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())

        model.add(Dense(num_classes))
        model.add(Activation('sigmoid'))

        def multilabel_accuracy(y_true, y_pred):
            return k.all(k.equal(y_true, k.round(y_pred)), axis=1)

        model.compile(loss='binary_crossentropy',
                      optimizer='sgd',
                      metrics=[multilabel_accuracy])

        return model
    
    def generate(self, generator_path, sample_size, key_mode, alphabet):
        (sample_array, keys_array, img_shape) = self.generate_text(generator_path=generator_path,
                                                                   sample_size=sample_size,
                                                                   key_mode=key_mode)
        training_size = int(sample_size * 0.8)
        training_array = sample_array[:training_size]
        test_array = sample_array[training_size:]

        (img_rows, img_cols) = img_shape

        channels_num = 1
        if k.image_data_format() == 'channels_first':
            training_array = training_array.reshape(training_array.shape[0], channels_num, img_rows, img_cols)
            test_array = test_array.reshape(test_array.shape[0], channels_num, img_rows, img_cols)
            input_shape = (channels_num, img_rows, img_cols)
        else:
            training_array = training_array.reshape(training_array.shape[0], img_rows, img_cols, channels_num)
            test_array = test_array.reshape(test_array.shape[0], img_rows, img_cols, channels_num)
            input_shape = (img_rows, img_cols, channels_num)

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

        from keras.utils import plot_model
        plot_model(model, show_shapes=True, to_file='model.png')

        model.fit(training_array, keys_training,
                  batch_size=16,
                  epochs=30,
                  verbose=2,
                  validation_data=(test_array, keys_test))

        model_result = model.evaluate(test_array, keys_test, verbose=0)
        return model, model_result
