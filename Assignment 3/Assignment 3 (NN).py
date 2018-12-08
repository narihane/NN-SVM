import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import validation_curve

# Hyper parameters
validation_size = 0.1
Lambda = 1e-05
batch_size = 64
epochs = 50

# [ (300), (400), (300, 300), (300, 400), (400, 300), (400, 400) ] 
param_range_HL = [x for x in itertools.product((300, 400),repeat=1)] + [x for x in itertools.product((300, 400),repeat=2)]
param_range_alpha = [0.00001, 0.001, 0.1, 1, 10]

grid_params_NN = {'hidden_layer_sizes': param_range_HL,
                  'alpha': param_range_alpha}

path = "../Datasets/UCI HAR Dataset/"

# Reading training data
X_train = np.loadtxt(path + "train/X_train.txt")
Y_train = np.loadtxt(path + "train/Y_train.txt")

# Reading test data
X_test = np.loadtxt(path + "test/X_test.txt")
Y_test = np.loadtxt(path + "test/Y_test.txt")


## preprocessing the data using PCA technique
##  n_components = [5, 50, 200, 500]
#pca = PCA(n_components=500)
#pca.fit(X_train)
#X_train = pca.transform(X_train)
#X_test = pca.transform(X_test)

train_scores, valid_scores = validation_curve(MLPClassifier(random_state=101), X_train, Y_train, 'hidden_layer_sizes', param_range_HL, cv=2, verbose=True, n_jobs=-1)

train_scores_mean = np.mean(train_scores, axis=1)
valid_scores_mean = np.mean(valid_scores, axis=1)

# Plotting the gradient error curves
plt.figure(1)
plt.plot(np.arange(0, len(train_scores_mean)), train_scores_mean, color='red', label='Training')
plt.plot(np.arange(0, len(valid_scores_mean)), valid_scores_mean, color='blue', label='Validation')

# setting title and labels
plt.title('Hidden layer ID VS Score')
plt.xlabel('Hidden layer ID')
plt.ylabel('Score')
plt.legend()

# displaying the plot
plt.show()

train_scores, valid_scores = validation_curve(MLPClassifier(random_state=101), X_train, Y_train, 'alpha', param_range_alpha, cv=2, verbose=True, n_jobs=-1)

train_scores_mean = np.mean(train_scores, axis=1)
valid_scores_mean = np.mean(valid_scores, axis=1)

# Plotting the gradient error curves
plt.figure(2)
plt.plot(param_range_alpha, train_scores_mean, color='red', label='Training')
plt.plot(param_range_alpha, valid_scores_mean, color='blue', label='Validation')

# setting title and labels
plt.title('Alpha VS Score')
plt.xlabel('Alpha')
plt.ylabel('Score')
plt.legend()

# displaying the plot
plt.show()

# Initializing the neural network classifier with the best parameters generated
# Best params hidden_layer_sizes=param_range_HL[1], alpha=1 without PCA
# Best params hidden_layer_sizes=param_range_HL[1], alpha=1 with PCA N = 5
# Best params hidden_layer_sizes=param_range_HL[3], alpha=0.01 with PCA N = 50
# Best params hidden_layer_sizes=param_range_HL[1], alpha=0.00001 with PCA N = 200
# Best params hidden_layer_sizes=param_range_HL[5], alpha=0.00001 with PCA N = 500
clf = MLPClassifier(hidden_layer_sizes=param_range_HL[1], alpha=0.00001, early_stopping=True, random_state=101)

# training the model
clf.fit(X_train, Y_train)

# Test the classifier
predictions = clf.predict(X_test)

# Print the various scores for the classifier
print(classification_report(Y_test, predictions))

# Plotting the gradient error curves
plt.figure(3)
plt.plot(np.arange(0, len(clf.loss_curve_)), clf.loss_curve_, color='red', label='Training')
plt.plot(np.arange(0, len(clf.validation_scores_)), 1 - np.array(clf.validation_scores_), color='blue', label='Validation')

# setting title and labels
plt.title('Number of iterations VS Errors')
plt.xlabel('Number of iterations')
plt.ylabel('Number of errors')
plt.legend()

# displaying the plot
plt.show()