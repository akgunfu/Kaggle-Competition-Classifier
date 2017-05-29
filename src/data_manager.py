import pandas as pd     # For Data Manipulation

from keras.utils import np_utils    # For Binarization
from sklearn.preprocessing import LabelEncoder  # For Encoding Labels to Integers


class data_manager():

    def __init__(self):
        # Read Test and Training Data From the Given Path
        self.train_df = pd.read_csv('../data/train.csv')
        self.test_df = pd.read_csv('../data/test.csv')

    # Return to Input X and Output Y
    def get_data(self):
        # Encoding Labels to Integer Values
        y = self.train_df['target'].values
        encoder = LabelEncoder()
        encoder.fit(y)
        encoded_y = encoder.transform(y)

        X = self.train_df.drop(['id', 'target'], axis = 1).values
        Y = np_utils.to_categorical(encoded_y).astype(int)  # Binarization of Label Values

        return (X,Y)