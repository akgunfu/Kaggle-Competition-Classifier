# Kaggle-Competition-Classifier
A multiclass classifier project to classify Kaggle competition data, Term Project of Learning from Data class Spring '17

1- Data Normalization
Data we have has 93 features and 9 classes. Each classes has name "Class_1, Class_2" etc. So in the first part,
in data_manager.py, we are dealing with changing data to a for which we can then use in any learning model. First we change
class labels to integers and then binarized them.

2- Learning Model