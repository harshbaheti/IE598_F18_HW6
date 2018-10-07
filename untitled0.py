from sklearn.tree import export_graphviz
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import time
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np


print( 'The scikit learn version is {}.'.format(sklearn.__version__))
Insample_score=[]
Out_of_sample_score=[]

start = time.clock()

 
  
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
   

for i in range(1,11):
    X_train, X_test, y_train, y_test = train_test_split(
           X, y, test_size=0.1, random_state=i)
    
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    
    def plot_decision_regions(X, y, classifier,
                           test_idx=None, resolution=0.02):
           
               
    
           markers = ('s', 'x', 'o', '^', 'v')
           colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
           cmap = ListedColormap(colors[:len(np.unique(y))])
           # plot the decision surface
           x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
           x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
           xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                                np.arange(x2_min, x2_max, resolution))
           Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
           Z = Z.reshape(xx1.shape)
           plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
           plt.xlim(xx1.min(), xx1.max())
           plt.ylim(xx2.min(), xx2.max())
           # plot all samples
           for idx, cl in enumerate(np.unique(y)):
               plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                           alpha=0.8, c=cmap(idx),
                           marker=markers[idx], label=cl)
           # highlight test samples
           if test_idx:
               X_test, y_test = X[test_idx, :], y[test_idx]
               plt.scatter(X_test[:, 0], X_test[:, 1], c='',
                           alpha=1.0, linewidths=1, marker='o',
                            s=55, label='test set')
               
               
               
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    
    
    tree = DecisionTreeClassifier(criterion='gini', 
                                  max_depth=4, 
                                  random_state=i)
    tree.fit(X_train, y_train)
    
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X_combined, y_combined, 
                          classifier=tree, test_idx=range(105, 150))
    
    plt.xlabel('petal length [cm]')
    plt.ylabel('petal width [cm]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    
    dot_data = export_graphviz(tree,
                               filled=True, 
                               rounded=True,
                               class_names=['Setosa', 
                                            'Versicolor',
                                            'Virginica'],
                               feature_names=['petal length', 
                                              'petal width'],
                               out_file=None) 
    y_test_pred = tree.predict(X_test)
    y_train_pred = tree.predict(X_train) 

    a = metrics.accuracy_score(y_train, y_train_pred)
    b = metrics.accuracy_score(y_test, y_test_pred)
    Insample_score.append(a)    
    Out_of_sample_score.append(b)
    plt.show(tree)
    print('\n\nGraph for Random State: %d, \nIn-sample score: %.3f,\nOut of sample score: %.3f' % (i,a,b))

print("\nInsample mean:\n",np.mean(Insample_score),"\nstandard deviation:\n",np.std(Insample_score))
print("\nOut_of_sample mean:\n",np.mean(Out_of_sample_score),"\nstandard deviation:\n",np.std(Out_of_sample_score))
print("\n")
end = time.clock()
print (end-start)
c = cross_val_score(estimator=tree,X=X,y=y,cv=10,n_jobs=1)
print('\nCV accuracy scores: %s' % c)
print('\nCV accuracy scores mean: %.3f ' % (np.mean(c))) 
print('\nstandard deviation: +/- %.3f \n' % (np.std(c))) 

print("My name is Harshavardhan Baheti")
print("My NetID is hbaheti3")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")