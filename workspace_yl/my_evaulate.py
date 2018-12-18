import bird_or_bicycle
import numpy as np
from keras import models
import pdb

def my_very_robust_model(images_batch_nhwc):
  """ This fn is a valid defense that always predicts the second class """
  batch_size = len(images_batch_nhwc)
  logits_np = np.array([[-5.0, 5.0]] * batch_size)
  return logits_np.astype(np.float32)

def test_model(X_test):
  model = models.load_model('result/temp_model.h5')
  pred = model.predict(X_test)
  return pred

# Evaluate the model (this will take ~10 hours on a GPU)
from unrestricted_advex import eval_kit
eval_kit.evaluate_bird_or_bicycle_model(test_model)
