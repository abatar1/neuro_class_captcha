import cv2
from keras.models import load_model
import numpy as np


class ModelPrediction:
    def __init__(self):
        pass

    @staticmethod
    def predict(img):
        if img is None:
            return None

        weight_dir_name = 'weights/'
        models_dir_name = 'models/'

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (img_rows, img_cols) = gray_img.shape
        gray_img = gray_img.reshape(1, img_rows, img_cols, 1)

        s_model = load_model(models_dir_name + 'num.h5')
        s_model.load_weights(weight_dir_name + 'num_weights.h5')
        size_prediction = s_model.predict(gray_img)
        size = np.argmax(size_prediction, axis=-1)

        t_model = load_model(models_dir_name + str(size[0]) + '.h5')
        t_model.load_weights(weight_dir_name + str(size[0]) + '_weights.h5')
        label_prediction = t_model.predict(gray_img)

        return size, np.argmax(label_prediction, axis=-1)
