import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv(
    filepath_or_buffer='http://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data', 
    header=None, 
    sep=',')

data = df.iloc[:,1:17]
target_names = df.iloc[:,17]
t_arr = [1,2,3,4,5,6,7]
newdata = data
colors = ['navy', 'turquoise', 'darkorange', 'red', 'yellow', 'green','pink']
lw=1

pca = PCA(n_components=2)
X_r = pca.fit(data).transform(data)
print('PCA explained variance ratio : %s' % str(pca.explained_variance_ratio_))

pipe_pca = Pipeline([('pca', pca),('tree', DecisionTreeClassifier())])
pipe_pca.fit(data, target_names)
pipe_pca.predict(newdata)
print ('PCA Accuracy: %s' % str( cross_val_score(pipe_pca, data, target_names)))

lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(data,target_names).transform(data)
print('LDA explained variance ratio : %s' % str(lda.explained_variance_ratio_))

pipe_lda = Pipeline([('lda', lda),('tree', DecisionTreeClassifier())])
pipe_lda.fit(data, target_names)
pipe_lda.predict(newdata)
print ('LDA Accuracy: %s' % str( cross_val_score(pipe_lda, data, target_names)))

for color, i, target_name in zip(colors, [1, 2,3,4,5,6,7,8], t_arr):
    plt.scatter(X_r[target_names == i, 0], X_r[target_names == i, 1], color=color, alpha=.8, lw=lw,label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of Zoo dataset')
plt.figure()

for color, i, target_name in zip(colors, [1, 2,3,4,5,6,7,8], t_arr):
    plt.scatter(X_r2[target_names == i, 0], X_r2[target_names == i, 1], color=color, alpha=.8, lw=lw,label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of Zoo dataset')