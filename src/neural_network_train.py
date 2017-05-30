import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from src.data_manager import data_manager

dm = data_manager()
X, Y, encoder = dm.get_data()
X_Test = dm.get_test_data()

seed = 7
np.random.seed(seed)

model = Sequential()
def network_model():
    model.add(Dense(100, input_dim=93, kernel_initializer='normal', activation='relu'))
    model.add(Dense(9, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=network_model, epochs=50, batch_size=20, verbose=1)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# Save Model to Disk
model_json = model.to_json()
with open("./models/keras_model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("./models/keras_weights.h5")


