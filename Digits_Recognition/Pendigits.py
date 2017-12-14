import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import time
from sklearn import preprocessing
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from scipy import cluster
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

heldout = [0.90, 0.80, 0.70, 0.60, 0.50]
time_arr=[]
accuracy_arr=[]
l=[]
time_arr2=[]
accuracy_arr2=[]
rounds = 20
df = pd.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tra', 
    header=None, 
    sep=',')

data = df.iloc[:,0:16]
target_names = df.iloc[:,16]
X, y = data, target_names

###############
pca = PCA(n_components=8)
X_pca = pca.fit(data).transform(data)
###############################
lda = LinearDiscriminantAnalysis(n_components=8, solver='svd')
X_lda = lda.fit(X, y).transform(X)
################################
#print("DIMENSIONALITY REDUCTION TECHNIQUES")
#pipe = Pipeline([
#    ('reduce_dim', PCA()),
#    ('classify', LinearSVC())
#])
#N_FEATURES_OPTIONS = [2, 4, 8]
#C_OPTIONS = [1, 10, 100, 1000]
#param_grid = [
#    {
#        'reduce_dim': [PCA(iterated_power=7),KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)],
#        'reduce_dim__n_components': N_FEATURES_OPTIONS,
#        'classify__C': C_OPTIONS
#    },
#]
#
#reducer_labels = ['PCA', 'KernalPCA']
#
#grid = GridSearchCV(pipe, cv=3, n_jobs=1, param_grid=param_grid)
#grid.fit(X, y)
#
#mean_scores = np.array(grid.cv_results_['mean_test_score'])
## scores are in the order of param_grid iteration, which is alphabetical
#mean_scores = mean_scores.reshape(len(C_OPTIONS), -1, len(N_FEATURES_OPTIONS))
## select score for best C
#mean_scores = mean_scores.max(axis=0)
#bar_offsets = (np.arange(len(N_FEATURES_OPTIONS)) *
#               (len(reducer_labels) + 1) + .5)
#plt.figure()
#COLORS = 'bry'
#for i, (label, reducer_scores) in enumerate(zip(reducer_labels, mean_scores)):
#    plt.bar(bar_offsets + i, reducer_scores, label=label, color=COLORS[i])
#
#plt.title("Comparing feature reduction techniques")
#plt.xlabel('Reduced number of features')
#plt.xticks(bar_offsets + len(reducer_labels) / 2, N_FEATURES_OPTIONS)
#plt.ylabel('Digits classification accuracy')
#plt.ylim((0, 1))
#plt.legend(loc='upper left')
#plt.show()

#Comparing several Classifiers on dataset WITHOUT DR
############
print("CLASSIFICATION TECHNIQUES")
classifiers = [
    ("SGD", SGDClassifier(), "aqua"),
    ("Perceptron", Perceptron(), "black"),
    ("SAG", LogisticRegression(solver='sag', tol=1e-1, C=1.e4 / X.shape[0]), "yellow"),
    ("SVM",SVC(kernel="linear", C=0.025), "orange"),
    ("KNeighbors",KNeighborsClassifier(3),"purple"),
    ("D-Tree",DecisionTreeClassifier(max_depth=5), "brown"),
    ("RF",RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),"green"),
    ("Naive Bayes",GaussianNB(),"red")
]

xx = 1. - np.array(heldout)
print("Classification directly on dataset without DR")
for name, clf,color in classifiers:
    print("Training %s" % name)
    rng = np.random.RandomState(42)
    yy = []
    
    for i in heldout:
        t0 = time()
        yy_ = []
        for r in range(rounds):
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=i, random_state=rng)
            
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            yy_.append(1 - np.mean(y_pred == y_test))
        time_=time()-t0
        time_arr.append(time_)
        accuracy_arr.append(accuracy_score(y_test,y_pred))
        yy.append(np.mean(yy_))
        print('Testing Accuracy: %f\tTime: %.2fs' % (accuracy_score(y_test,y_pred),time_))

    y_new = [yy * 100 for yy in yy]
    plt.plot(xx, y_new, label=name, color=color)

my_xticks = ['10%','20%','30%','40%','50%']
plt.xticks(xx, my_xticks)
plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)
plt.xlabel("Training data %")
plt.ylabel("Test Error %")
plt.title("Comparison of Classifiers without DR")
plt.show()
#
## Accuracy comparison graph
sgd=accuracy_arr[0:5]
perc=accuracy_arr[5:10]
sag=accuracy_arr[10:15]
svm=accuracy_arr[15:20]
knn=accuracy_arr[20:25]
dtree=accuracy_arr[25:30]
rf=accuracy_arr[30:35]
nb=accuracy_arr[35:40]

acc=[("SGD",sgd,"aqua"),
     ("perceptron",perc,"black"),
     ("SAG", sag,"yellow"),
     ("SVM",svm,"orange"),
     ("KNeighbors",knn,"purple"),
     ("D-Tree",dtree,"brown"),
     ("RF",rf,"green"),
     ("Naive Bayes",nb,"red")
     ]
for name, class_,color in acc:
    l = [class_ * 100 for class_ in class_]
    plt.plot(xx,l, label=name, color=color)

my_xticks = ['10%','20%','30%','40%','50%']
plt.xticks(xx, my_xticks)
plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)
plt.xlabel("Training data %")
plt.ylabel("Accuracy %")
plt.title("Accuracy Comparison of Classifiers without DR")
plt.show()
#################
# Time comparison graph
sgd_t=time_arr[0:5]
perc_t=time_arr[5:10]
sag_t=time_arr[10:15]
svm_t=time_arr[15:20]
knn_t=time_arr[20:25]
dtree_t=time_arr[25:30]
rf_t=time_arr[30:35]
nb_t=time_arr[35:40]

t=[("SGD",sgd_t,"aqua"),
     ("perceptron",perc_t,"black"),
     ("SAG", sag_t,"yellow"),
     ("SVM",svm_t,"orange"),
     ("KNeighbors",knn_t,"purple"),
     ("D-Tree",dtree_t,"brown"),
     ("RF",rf_t,"green"),
     ("Naive Bayes",nb_t,"red")
     ]
for name, class_,color in t:
    plt.plot(xx,class_, label=name, color=color)

my_xticks = ['10%','20%','30%','40%','50%']
plt.xticks(xx, my_xticks)
plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)
plt.xlabel("Training data %")
plt.ylabel("Time in seconds")
plt.title("Time Comparison of Classifiers without DR")
plt.show()

#Comparing several Classifiers on dataset WITH DR
############
classifiers = [
    ("SGD", SGDClassifier(), "aqua"),
    ("Perceptron", Perceptron(), "black"),
    ("SAG", LogisticRegression(solver='sag', tol=1e-1, C=1.e4 / X_lda.shape[0]), "yellow"),
    ("SVM",SVC(kernel="linear", C=0.025), "orange"),
    ("KNeighbors",KNeighborsClassifier(3),"purple"),
    ("D-Tree",DecisionTreeClassifier(max_depth=5), "brown"),
    ("RF",RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),"green"),
    ("Naive Bayes",GaussianNB(),"red")
]

xx = 1. - np.array(heldout)
print("Classification directly on dataset with DR")
for name, clf,color in classifiers:
    print("Training %s" % name)
    rng = np.random.RandomState(42)
    yy = []
    yy2 = []
    
    for i in heldout:
        t0 = time()
        yy_ = []
        yy2_ = []
        for r in range(rounds):
            X_train, X_test, y_train, y_test = \
                train_test_split(X_lda, y, test_size=i, random_state=rng)
            
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            yy2_.append(1 - np.mean(y_pred == y_test))
        time_=time()-t0
        time_arr2.append(time_)
        accuracy_arr2.append(accuracy_score(y_test,y_pred))
        yy2.append(np.mean(yy2_))
        print('Testing Accuracy: %f\tTime: %.2fs' % (accuracy_score(y_test,y_pred),time_))

    y_new = [yy2 * 100 for yy2 in yy2]
    plt.plot(xx, y_new, label=name, color=color)

my_xticks = ['10%','20%','30%','40%','50%']
plt.xticks(xx, my_xticks)
plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)
plt.xlabel("Training data %")
plt.ylabel("Test Error %")
plt.title("Comparison of Classifiers with DR")
plt.show()

# Accuracy comparison graph
sgd=accuracy_arr2[0:5]
perc=accuracy_arr2[5:10]
sag=accuracy_arr2[10:15]
svm=accuracy_arr2[15:20]
knn=accuracy_arr2[20:25]
dtree=accuracy_arr2[25:30]
rf=accuracy_arr2[30:35]
nb=accuracy_arr2[35:40]

acc=[("SGD",sgd,"aqua"),
     ("perceptron",perc,"black"),
     ("SAG", sag,"yellow"),
     ("SVM",svm,"orange"),
     ("K-NN",knn,"purple"),
     ("D-Tree",dtree,"brown"),
     ("RF",rf,"green"),
     ("Naive Bayes",nb,"red")
     ]
for name, class_,color in acc:
    l = [class_ * 100 for class_ in class_]
    plt.plot(xx,l, label=name, color=color)

my_xticks = ['10%','20%','30%','40%','50%']
plt.xticks(xx, my_xticks)
plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)
plt.xlabel("Training data %")
plt.ylabel("Accuracy %")
plt.title("Accuracy Comparison of Classifiers with DR")
plt.show()
#################
# Time comparison graph
sgd_t=time_arr2[0:5]
perc_t=time_arr2[5:10]
sag_t=time_arr2[10:15]
svm_t=time_arr2[15:20]
knn_t=time_arr2[20:25]
dtree_t=time_arr2[25:30]
rf_t=time_arr2[30:35]
nb_t=time_arr2[35:40]

t=[("SGD",sgd_t,"aqua"),
     ("perceptron",perc_t,"black"),
     ("SAG", sag_t,"yellow"),
     ("SVM",svm_t,"orange"),
     ("K-NN",knn_t,"purple"),
     ("D-Tree",dtree_t,"brown"),
     ("RF",rf_t,"green"),
     ("Naive Bayes",nb_t,"red")
     ]
for name, class_,color in t:
    plt.plot(xx,class_, label=name, color=color)

my_xticks = ['10%','20%','30%','40%','50%']
plt.xticks(xx, my_xticks)
plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)
plt.xlabel("Training data %")
plt.ylabel("Time in seconds")
plt.title("Time Comparison of Classifiers with DR")
plt.show()

## Clustering using KMeans
################
#print("CLUSTERING TECHNIQUES")
#np.random.seed(42)
#
#df = pd.read_csv(
#    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tra', 
#    header=None, 
#    sep=',')
#
#data = df.iloc[:,0:16]
#target = df.iloc[:,16]
#data = scale(data)
#
#n_samples, n_features = data.shape
#n_digits = len(np.unique(target))
#labels = target
#
#sample_size = len(df.index)
#
#print("n_digits: %d, \t n_samples %d, \t n_features %d"
#      % (n_digits, n_samples, n_features))
#
##Elbow graph
##plot variance for each value for 'k' between 1,16
#initial = [cluster.vq.kmeans(data,i) for i in range(1,16)]
#plt.plot([var for (cent,var) in initial])
#plt.xlabel("Number of clusters")
#plt.ylabel("Mean squared error")
#plt.title("Elbow graph")
#plt.show()
#
#print(82 * '_')
#print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')
#
#def bench_k_means(estimator, name, data):
#    t0 = time()
#    estimator.fit(data)
#    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
#          % (name, (time() - t0), estimator.inertia_,
#             metrics.homogeneity_score(labels, estimator.labels_),
#             metrics.completeness_score(labels, estimator.labels_),
#             metrics.v_measure_score(labels, estimator.labels_),
#             metrics.adjusted_rand_score(labels, estimator.labels_),
#             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
#             metrics.silhouette_score(data, estimator.labels_,
#                                      metric='euclidean',
#                                      sample_size=sample_size)))
#
#bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
#              name="k-means++", data=data)
#
#bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),
#              name="random", data=data)
#
## in this case the seeding of the centers is deterministic, hence we run the
## kmeans algorithm only once with n_init=1
#pca = PCA(n_components=n_digits).fit(data)
#bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
#              name="PCA-based",
#              data=data)
#print(82 * '_')
#
## #############################################################################
## Visualize the results on PCA-reduced data
#
#reduced_data = PCA(n_components=2).fit_transform(data)
#kmeans = KMeans(init='k-means++', n_clusters=8, n_init=10)
#kmeans.fit(reduced_data)
#
## Step size of the mesh. Decrease to increase the quality of the VQ.
#h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].
#
## Plot the decision boundary. For that, we will assign a color to each
#x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
#y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
#xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#
## Obtain labels for each point in mesh. Use last trained model.
#Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
#
## Put the result into a color plot
#Z = Z.reshape(xx.shape)
#plt.figure(1)
#plt.clf()
#plt.imshow(Z, interpolation='nearest',
#           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
#           cmap=plt.cm.Paired,
#           aspect='auto', origin='lower')
#
#plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
## Plot the centroids as a white X
#centroids = kmeans.cluster_centers_
#plt.scatter(centroids[:, 0], centroids[:, 1],
#            marker='x', s=169, linewidths=3,
#            color='w', zorder=10)
#plt.title('K-means clustering on the digits dataset (PCA-reduced data)')
#plt.xlim(x_min, x_max)
#plt.ylim(y_min, y_max)
#plt.xticks(())
#plt.yticks(())
#plt.show()