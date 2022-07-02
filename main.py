"""
                    Abdullah Baig  - 231485698
                    Sameed Gillani - 231488347
                    Haseeb Asif    - 231492733
"""

# All the functions used in main.py are imported from utils.py

from utils import *
from utils import _train_test_split

le = preprocessing.LabelEncoder()

if __name__ == "__main__":
    df = pd.read_csv('data.csv')  # Importing the data.csv file

    x = df.drop(columns=["in_college"], axis=1)  # Dropping the y_table (outcome)
    x = x.apply(le.fit_transform)       # Encoding the string variables to numerical couterparts

    y = df.filter(["in_college"])  # Dropping everything except the y_table (outcome)
    y = y.apply(le.fit_transform)  # Encoding the string variables to numerical couterparts

    x_train, x_test, y_train, y_test = _train_test_split(x, y)  # Split the data into training and testing parts

    predicted = knn_classifier(x_test, x_train, y_test, y_train)  # Predicted outcome for the given testing data
    y_test = list(map(lambda x: list(x)[0], list(y_test.values)))  # Unpacking the predicted data for confusion_matrix

    print()
    print("Confusion Matrix: \n", get_confusion_matrix(y_test, predicted))
    print("\nAccuracy of the Model: ", get_accuracy(y_test, predicted), "%")
    print("\nPrecision of the Model: ", get_precision(y_test, predicted), "%")
    print("\nF-Score of the Model: ", get_f1_score(y_test, predicted), "%")

