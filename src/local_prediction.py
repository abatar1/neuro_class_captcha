import model_prediction as mp
import php_helper as ph

import subprocess
import cv2
import numpy as np

from matplotlib import pyplot as plt

helper = ph.PhpHelper()
helper.set_length((1, 8))

generator_path='kcaptcha/index.php'
s = subprocess.check_output(['php', generator_path])
img = cv2.imdecode(np.frombuffer(s, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
plt.imshow(img)
plt.show()

predict = mp.ModelPrediction()
size, prediction = predict.predict(img)

print('size: ' + ''.join(str(e) for e in size))
print('prediction: ' + ''.join(str(e) for e in prediction))
