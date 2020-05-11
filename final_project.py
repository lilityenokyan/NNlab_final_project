import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

#STEP 1
names = ["age", "year", "num", "res"]
data = pd.read_csv('C:/Users/Lilit/Downloads/haberman.data', sep=",", header=None, names = names)

X = data[['age', 'num']]
Y = data[["res"]]

#PLOT the graph of data using the age and the # of positive axillary nodes
#grouped by the survivial status: survived-red, died-blue

colors= ['red' if l == 1 else 'blue' for l in Y["res"]]
plt.scatter(data["age"],data["num"],color = colors)

# Give labels to the axis of the graph as well as a title
plt.xlabel("Patient's age")
plt.ylabel("# of positive axillary nodes")
plt.title("Haberman's Survival Data Representation")
plt.show()

#STEP 2

min_max_scaler = preprocessing.MinMaxScaler()

X =  min_max_scaler.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)

#STEP 3

rate = 0.3
maxEpoch = 1000
pg = 1e-5

clf = MLPClassifier(hidden_layer_sizes = (15, 15), activation = "logistic", solver = "lbfgs", learning_rate_init = rate, max_iter = maxEpoch, tol = pg, early_stopping = True) 

clf.fit(X_train,Y_train)

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# step size in the mesh
h = .001

# Plot the decision boundary. 
# Mesh is of size [a_min, a_max]x[b_min, b_max] where a and b are two features displayed on two axis of the plot
# Each point in the mesh has a color for some reason

a_min, a_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
b_min, b_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
aa, bb = np.meshgrid(np.arange(a_min, a_max, h),
                        np.arange(b_min, b_max, h))
                        
# Make couples from two feature sets for prediction 
features = np.c_[aa.ravel(), bb.ravel()]

# Put predictions in the Z variable to use it as a third coordinate to group/color the points
# Put the result into a color plot
Z = clf.predict(features)
Z = Z.reshape(aa.shape)

# Put predictions in the Z variable to use it as a third coordinate to group/color the points
pl.figure()
pl.pcolormesh(aa, bb, Z, cmap=cmap_light)

# Add the training points to the mesh
pl.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap=cmap_bold)

# Predict the results using the test set
y_predicted = clf.predict(X_test)

# Add the testing points to the mesh
pl.scatter(X_test[:, 0], X_test[:, 1], c=y_predicted, alpha=0.1, cmap=cmap_bold)

# Give the plot dimensions
pl.xlim(aa.min(), aa.max())
pl.ylim(bb.min(), bb.max())

# Give labels to the axis of the graph as well as a title
pl.xlabel("Patient's age (transformed)")
pl.ylabel("# of positive axillary nodes (transformed)")
pl.title("Classification plot obtaied by the ANN")

# Show the plot
pl.show()

# Get the accuracy score for the prediction and print it
score = clf.score(X_test, Y_test)
print(score * 100)

# Get the confusion matrix and print it
conf_matrix = confusion_matrix(Y_test,y_predicted)
print(conf_matrix)
