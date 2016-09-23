"""
This is just some testing for farming out the work of regression, where it applies,
during experiments. This script just wraps various python libraries for performing
logistic and other forms of regression, as might be useful for learning the weights
in the reward function, given an input set of terminal-state prototypes (see prototypes.csv).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

# import some data to play with
prototypes = np.loadtxt("prototypes_BACKUP.csv",delimiter=",")
#X = prototypes.data[:, :3]  # we only take the first two features.
#Y = prototypes.target
X = prototypes[:,0:3] #all rows, cols 0-2 
Y = prototypes[:,3]

print("xs:\n"+str(X))
print("ys:\n"+str(Y))


#h = .02  # step size in the mesh

logreg = linear_model.LogisticRegression(C=1e5)

# we create an instance of Neighbours Classifier and fit the data.
logreg.fit(X, Y)

print(str(logreg.get_params()))
print(str(logreg.coef_))


"""
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('')
plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()
"""



