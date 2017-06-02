import pandas as pd     # For Data Manipulation

from keras.utils import np_utils    # For Binarization
from sklearn.preprocessing import LabelEncoder  # For Encoding Labels to Integers
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


class data_manager():

    def __init__(self):
        # Read Test and Training Data From the Given Path
        self.train_df = pd.read_csv('../data/train.csv')
        self.test_df = pd.read_csv('../data/test.csv')
        self.reduce = 90

    # Return to Input X and Output Y
    def get_data(self):
        # Encoding Labels to Integer Values
        y = self.train_df['target'].values
        encoder = LabelEncoder()
        encoder.fit(y)
        encoded_y = encoder.transform(y)
        Y = np_utils.to_categorical(encoded_y).astype(int)  # Binarization of Label Values

        X = self.train_df.drop(['id', 'target'], axis = 1).values
        X = scale(X)
        pca = PCA(n_components=self.reduce)
        pca.fit(X)
        X_pca = pca.fit_transform(X)

        return (X_pca, X, Y, encoder)

    # Return to Test Data
    def get_test_data(self):
        X_test = self.test_df.drop('id', axis = 1).values
        X_test = scale(X_test)
        pca = PCA(n_components=self.reduce)
        pca.fit(X_test)
        X_test_pca = pca.fit_transform(X_test)
        return (X_test_pca, X_test)