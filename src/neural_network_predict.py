from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from src.data_manager import data_manager
import numpy as np

dm = data_manager()
X_Test = dm.get_test_data()
_, _, encoder = dm.get_data()

json_file = open('./models/keras_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("./models/keras_weights.h5")

predictions_raw = model.predict(X_Test)
encoded_labels = np.argmax(predictions_raw, axis = 1)
decoded = encoder.inverse_transform(encoded_labels)

print (decoded)
print (np.unique(decoded))