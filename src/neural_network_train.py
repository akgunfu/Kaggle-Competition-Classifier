from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

import numpy as np

from src.data_manager import data_manager

# Getting X,Y data and Encoder for Labels
dm = data_manager()
X, Y, encoder = dm.get_data()

# Random State Seed
seed = 7
np.random.seed(seed)

# A function to build a model, model is initialized outside of function to be able to reference model from anywhere
model = Sequential()
def network_model():
    model.add(Dense(30, input_dim=30, kernel_initializer='normal', activation='sigmoid'))
    model.add(Dense(9, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Set up Estimator and Train System, Evaluate with KFold Cross Validation
estimator = KerasClassifier(build_fn=network_model, epochs=15, batch_size=30, verbose=1)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# Save Model to Disk
model_json = model.to_json()
with open("./models/keras_model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("./models/keras_weights.h5")