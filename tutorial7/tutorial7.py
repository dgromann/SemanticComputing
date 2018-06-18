import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score

from keras.models import Sequential
from keras.layers import Dense

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


#Load the wine datasets for white and red Portugese Vinho Verde
white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')
red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')

#Quick look at the data
print(white.info())
print(red.info())

print("First row of red ")
print(red.head(1))
print("Last row of white ")
print(white.tail(1))
print("Sample of 5 rows of red ", red.sample(5))
print("Description of white ", white.describe())


# Add `type` column to `red` with value 1
red['type'] = 1
# Add `type` column to `white` with value 0
white['type'] = 0
# Append `white` to `red` without keeping the index of white (double index)
wines = red.append(white, ignore_index=True)

print(wines.groupby('type').count())


#Plot the
corr = wines.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.title("Correlation Matrix")
plt.show()


# Specify the data
X = wines.ix[:,0:11]

# Specify the target labels and flatten the array
y = np.ravel(wines.type)

# Split the data up in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


# Standardize the data (to deal with values that lie far apart)
# Define the scaler
scaler = StandardScaler().fit(X_train)

# Scale the train set
X_train = scaler.transform(X_train)

# Scale the test set
X_test = scaler.transform(X_test)



#TODO: initialize the model with one input layer, one hiddne layer and one output layer (the last one has output shape 1 since it is a binary classifiction task)
# Initialize the constructor

# Add an input layer

# Add one hidden layer

# Add an output layer

# Model summary


#TODO: compile and fit the model and get the predictions
#Compile the model using binary_crossentropy, sgd, and metrics=['accuracy']

#Fit the model
y_pred = model.predict(X_test)


#Evaluation of the model
score = model.evaluate(X_test, y_test, verbose=1)
print("Loss and accuracy score", score)

print(y_pred)
print(y_pred.round())
# Confusion matrix
print(confusion_matrix(y_test, y_pred.round()))
plot_confusion_matrix(confusion_matrix(y_test, y_pred.round()), classes=["red", "white"],title='Confusion matrix, without normalization')
plt.show()
print(precision_score(y_test, y_pred.round()))
print(recall_score(y_test, y_pred.round()))
print(f1_score(y_test,y_pred.round()))
print(cohen_kappa_score(y_test, y_pred.round()))