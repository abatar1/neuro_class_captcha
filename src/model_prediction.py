import cv2
import numpy as np

class ModelPrediction:
    def predict(self, filename):
        img = np.array(cv2.imread(filename, cv2.IMREAD_GRAYSCALE))

        if img is None:
            print 'Wrong filename error!'
            return None

        from keras.models import load_model

        s_model = load_model('num.h5')
        s_model.load_weights('num_weights.h5')
        size_prediction = s_model.predict(img)

        t_model = load_model(size_prediction + '.h5')
        t_model.load_weights(size_prediction + '_weights.h5')
        return t_model.predict(img)