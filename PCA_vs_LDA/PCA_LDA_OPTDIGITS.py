import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score
from mpl_toolkits.mplot3d import Axes3D

optdigits = datasets.load_digits()
X = optdigits.data
y = optdigits.target
target_names = optdigits.target_names
newdata = optdigits.data
colors = ['red', 'turquoise', 'blue', 'green', 'pink', 'yellow', 'orange', 'purple', 'magenta','violet'] 
lw = 1

pca = PCA(n_components=5)
X_r = pca.fit(X).transform(X) #fill model with X and apply DR on X

lda = LinearDiscriminantAnalysis(n_components=5, solver='svd')
X_r2 = lda.fit(X, y).transform(X)

# Percentage of variance explained for each components
print('PCA explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))
print('LDA explained variance ratio (first two components): %s'
      % str(lda.explained_variance_ratio_))

pipe_pca = Pipeline([('pca', pca),('tree', DecisionTreeClassifier())])
pipe_pca.fit(optdigits.data, optdigits.target)
pipe_pca.predict(newdata)
print ('PCA Accuracy: %s' % str( cross_val_score(pipe_pca, optdigits.data, optdigits.target)))

pipe_lda = Pipeline([('lda', lda),('tree', DecisionTreeClassifier())])
pipe_lda.fit(optdigits.data, optdigits.target)
pipe_lda.predict(newdata)
print ('LDA Accuracy: %s' % str( cross_val_score(pipe_lda, optdigits.data, optdigits.target)))

plt.figure(figsize=(10,8))
for color, i, target_name in zip(colors, [0, 1, 2,3,4,5,6,7,8,9,10], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1],color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA (n_components=5, n_classes=10)')

plt.figure(figsize=(10,8))
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA (n_components=5, n_classes=3)')

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
for color, i, target_name in zip(colors, [0, 1, 2,3,4,5,6,7,8,9,10], target_names):
    ax.scatter(xs=X_r[y == i, 0], ys=X_r[y == i, 1], zs=X_r[y == i, 1], zdir='z', c=None, depthshade=True)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.figure(figsize=(10,8))
for color, i, target_name in zip(colors, [0, 1, 2,3,4,5,6,7,8,9,10], target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA (n_components=3, n_classes=10)')

plt.figure(figsize=(10,8))
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA (n_components=3, n_classes=3)')
plt.show()

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
for color, i, target_name in zip(colors, [0, 1, 2,3,4,5,6,7,8,9,10], target_names):
    ax.scatter(xs=X_r2[y == i, 0], ys=X_r2[y == i, 1], zs=X_r2[y == i, 1], zdir='z', c=None, depthshade=True)
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')