import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import validation_curve

# Hyper parameters
validation_size = 0.1
Lambda = 1e-05
batch_size = 64
epochs = 50

param_range_C = [0.1, 1, 10, 100, 1000]
param_range_gamma = [1, 0.1, 0.01, 0.001, 0.0001]

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

train_scores, valid_scores = validation_curve(SVC(random_state=101), X_train, Y_train, 'C', param_range_C, cv=2, verbose=True, n_jobs=-1)

train_scores_mean = np.mean(train_scores, axis=1)
valid_scores_mean = np.mean(valid_scores, axis=1)

# Plotting the gradient error curves
plt.figure(1)
plt.plot(param_range_C, train_scores_mean, color='red', label='Training')
plt.plot(param_range_C, valid_scores_mean, color='blue', label='Validation')

# setting title and labels
plt.title('C VS Score')
plt.xlabel('C')
plt.ylabel('Score')
plt.legend()

# displaying the plot
plt.show()

train_scores, valid_scores = validation_curve(SVC(random_state=101), X_train, Y_train, 'gamma', param_range_gamma, cv=2, verbose=True, n_jobs=-1)

train_scores_mean = np.mean(train_scores, axis=1)
valid_scores_mean = np.mean(valid_scores, axis=1)

# Plotting the gradient error curves
plt.figure(2)
plt.plot(param_range_gamma, train_scores_mean, color='red', label='Training')
plt.plot(param_range_gamma, valid_scores_mean, color='blue', label='Validation')

# setting title and labels
plt.title('gamma VS Score')
plt.xlabel('gamma')
plt.ylabel('Score')
plt.legend()

# displaying the plot
plt.show()

# Initializing the neural network classifier with the best parameters generated
# Best params C=1000, gamma=0.01 without PCA
# Best params C=0.1, gamma=0.01 with PCA N = 5
# Best params C=10, gamma=0.01 with PCA N = 50
# Best params C=100, gamma=0.01 with PCA N = 200
# Best params C=100, gamma=0.01 with PCA N = 500
svc = SVC(C=1000, gamma=0.01, random_state=101)

# training the model
svc.fit(X_train, Y_train)

# Test the classifier
predictions = svc.predict(X_test)

# Print the various scores for the classifier
print(classification_report(Y_test, predictions))