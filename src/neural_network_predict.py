from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np

from src.data_manager import data_manager

# Getting Test data and Label Encoder
dm = data_manager()
X_Test = dm.get_test_data()
_, _, encoder_y = dm.get_data()

# Load model from file
json_file = open('./models/keras_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("./models/keras_weights.h5")

# Make Predictions for Test Data, Decode Them Into String Classes, Encode to Binary Matrix Again to Print
predictions_raw = model.predict(X_Test)
encoded_labels = np.argmax(predictions_raw, axis = 1)
decoded = encoder_y.inverse_transform(encoded_labels)
encoder = LabelEncoder()
encoder.fit(decoded)
encoded_y = encoder.transform(decoded)
final = np_utils.to_categorical(encoded_y).astype(int)

# Create a Pandas.DataFrame to Represent Data
N = final.shape[0]
index = np.arange(0,N,1)
id = index + 1
columns = (['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9'])

dataframe = pd.DataFrame(data=final, index=index, columns = columns)
dataframe['id'] = pd.Series(id, index=dataframe.index)

# Rearrange Columns
cols = dataframe.columns.tolist()
cols = cols[-1:] + cols[:-1]
dataframe = dataframe[cols]

# Save DataFrame as Final Submission
filename = './submissions/nn_submission.csv'
dataframe.to_csv(filename, index=False, encoding='utf-8')

