---
layout: post
title:  "Scikit Learn Cheatsheet"
date:   2020-03-01 11:25:30 -0000
categories: machine learning
permalink: /machine-learning/SklearnCheatsheet.html
---

Scikit-learn (Sklearn) is the most useful and robust library for machine learning in Python. It provides a selection of efficient tools for machine learning and statistical modeling including classification, regression, clustering and dimensionality reduction via a consistence interface in Python. This library, which is largely written in Python, is built upon NumPy, SciPy and Matplotlib.  I write this post as a Scikit-Learn Cheatsheet.



## Step 1: Preprocessing the data

**Loading the data** 
Data needs to be numeric and stored as NumPy arrays or SciPy sparse matrices. Other types that are convertible to numeric arrays, such as Pandas DataFrame, are also acceptable. 
- Features − The variables of data are called its features. They are also known as predictors, inputs or attributes.
    - Feature matrix − It is the collection of features, in case there are more than one.  
    - Feature Names − It is the list of all the names of the features.

- Target − It is the output variable that basically depends upon the feature variables. They are also known as Response, label or output.
    - Target Vector − It is used to represent response column. Generally, we have just one response column. 
    - Target Names − It represent the possible values taken by a response vector. 
    
As we are dealing with lots of data and that data is in raw form, before inputting that data to machine learning algorithms, we need to convert it into meaningful data. This process is called preprocessing the data. Scikit-learn has package named preprocessing for this purpose. The preprocessing package has the following techniques 
    
### 1. Data Preprocessing

**Imputing Missing Values**
```
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values=0, strategy='mean', axis=0)
imp.fit_transform(X_train)
```
**Generating Polynomial Features**   
Polynomial features are often created when we want to include the notion that there exists a nonlinear relationship between the features and the target.  
```
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=3, interaction_only=True)
polynomials = poly.fit_transform(X)
```
**Encoding Categorical Features**
it is necessary to convert categorical features to a numerical representation.  
```
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
y = enc.fit_transform(y)
```
**Numerical features**  
Numerical features can be ‘decoded’ into categorical features. The two most common ways to do this are *discretization* and *binarization*.  
```
from sklearn.preprocessing import Binarizer
binarizer = Binarizer(threshold=0.0).fit(X)
binary_X = binarizer.transform(X)
```
**Custom transformers**  
We can implement a transformer from an arbitrary function with FunctionTransformer.  
```
from sklearn.preprocessing import FunctionTransformer
transformer = FunctionTransformer(np.log1p, validate=True)
transformer.fit_transform(X.f2.values.reshape(-1, 1))
```
**Feature scaling**  
sklearn provides a bunch of scalers:*StandardScaler, MinMaxScaler, MaxAbsScaler and RobustScaler*.  
```
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
standardized_X = scaler.transform(X_train)
standardized_X_test = scaler.transform(X_test)
```
**Normalization**  
Normalization is the process of scaling individual samples to have unit norm. Scaling inputs to unit norms is a common operation for text classification or clustering. Sklearn provides three norms are: l1, l2 and max.  
```
from sklearn.preprocessing import Normalizer
scaler = Normalizer().fit(X_train)
normalized_X = scaler.transform(X_train)
normalized_X_test = scaler.transform(X_test)
```
**Training-Test-Data split**
```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 random_state=0)
```

### 2. Feature selection

**Univariate Selection**  
Statistical tests can be used to select those features that have the strongest relationship with the output variable.  
```
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
test = SelectKBest(score_func=f_classif, k=4)
fit = test.fit(X, y)
print(fit.scores_)
```
**Recursive Feature Elimination**  
It uses the model accuracy to identify which attributes (and combination of attributes) contribute the most to predicting the target attribute.  
```
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs')
rfe = RFE(model, 3)
fit = rfe.fit(X, y)
print("Selected Features: %s" % fit.support_)
```
**Principal Component Analysis**  
Principal Component Analysis (or PCA) uses linear algebra to transform the dataset into a compressed form.  We can choose the number of dimensions or principal component in the transformed result.  
```
from sklearn.decomposition import PCA
import scikitplot as skplt
pca = PCA(n_components=3)
fit = pca.fit(X)
print("Explained Variance: %s" % fit.explained_variance_ratio_)
skplt.decomposition.plot_pca_component_variance(pca)
```
**Feature Importance**  
Bagged decision trees like Random Forest and Extra Trees can be used to estimate the importance of features.  
```
from sklearn.ensemble import ExtraTreesClassifier
import scikitplot as skplt
model = ExtraTreesClassifier(n_estimators=10)
model.fit(X, y)
print(model.feature_importances_)
skplt.estimators.plot_feature_importances(model, feature_names=['col1', 'col2', 'col3'])
```



## Step 2: Create  model, Fit and Prediction  

  
The flowchart below is designed by Scikit-learn to give users a bit of a rough guide on how to approach problems with regard to which estimators to try on our data. 

![Choosing the right estimator](https://scikit-learn.org/stable/_static/ml_map.png)
 
 
### 1. Regression


**Linear Regression**  

Import and create the model: 
```
from sklearn.linear_model import LinearRegression   
lr = LinearRegression(normalize=True)
```
Fit:
```
lr.fit(X_train, y_train)
```
`.coef_`: contains the coefficients  
`.intercept`_: contains the intercept  
Predict:
```
y_pred = lr.predict(X_test)
```
`.score()`: returns the coefficient of determination R²  

**Ridge Regression**  
Ridge regression is the regularization technique that performs L2 regularization. It modifies the loss function by adding the penalty equivalent to the square of the magnitude of coefficients.  
```
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=0.01)
ridge.fit(X_train, y_train) 
y_pred = ridge.predict(X_test)
```
**LASSO**  
LASSO is the regularisation technique that performs L1 regularisation. It modifies the loss function by adding the penalty equivalent to the summation of the absolute value of coefficients. Lasso Regression can also be used for feature selection because the coeﬃcients of less important features are reduced to zero.
```
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train) 
y_pred = lasso.predict(X_test)
```
**Polynomial Regression**  
A simple linear regression can be extended by constructing polynomial features from the coefficients.  
```
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
model = Pipeline([('poly', PolynomialFeatures(degree=3)), ('linear', LinearRegression(fit_intercept=False))])
model.fit(X_train, y_train)
model.named_steps['linear'].coef_
```

### 2. Classification  

**Logistic Regression**  
Logistic regression, despite its name, is a classification algorithm rather than regression algorithm. Based on a given set of independent variables, it is used to estimate discrete value (0 or 1, yes/no, true/false).   
```
from sklearn.linear_model import LogisticRegression
LRG = LogisticRegression(random_state = 0,solver = 'liblinear',multi class = 'auto')
LRG.fit(X, y)
LRG.score(X, y)
````
**Support Vector Machines (SVM)**  
SVM is used for both classification and regression problems. Scikit-learn provides three classes namely SVC, NuSVC and LinearSVC which can perform multiclass-class classification.
```
from sklearn.svm import SVC
svc = SVC(kernel = 'linear',gamma = 'scale', shrinking = False)
svc_Clf.fit(X, y)
y_pred = svc.predict(np.random.random((2,5)))
```
**Naive Bayes**  
Naïve Bayes methods are a set of supervised learning algorithms based on applying Bayes’ theorem with a strong assumption that all the predictors are independent to each other. The Scikit-learn provides different naïve Bayes classifiers models namely Gaussian, Multinomial, Complement and Bernoulli. All of them differ mainly by the assumption they make regarding the distribution of P. the probability of predictor given class.
```
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
```
**K-Nearest Neighbors**  
k-NN (k-Nearest Neighbor), one of the simplest machine learning algorithms, is non-parametric and lazy in nature.  
```
from sklearn.neigbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict_proba(X_test)
y_proba = knn.predict_proba(X_test)
```
**Decision Tree**  
Decisions trees are the most powerful algorithms that falls under the category of supervised algorithms. They can be used for both classification and regression tasks. 
```
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
```

### 3. Ensemble Learning

**Random Forest**  
In Random forest, each decision tree in the ensemble is built from a sample drawn with replacement from the training set and then gets the prediction from each of them and finally selects the best solution by means of voting. It can be used for both classification as well as regression tasks.
```
from sklearn.ensemble import RandomForestClassifier
RFclf = RandomForestClassifier(n_estimators = 50)
RFclf.fit(X_train, y_train)
y_pred = RFclf.predict(X_test)
```

**AdaBoost**  
Boosting methods build ensemble model in an increment way. The main principle is to build the model incrementally by training each base model estimator sequentially. In order to build powerful ensemble, these methods basically combine several week learners which are sequentially trained over multiple iterations of training data.   
AdaBoost is one of the most successful boosting ensemble method whose main key is in the way they give weights to the instances in dataset. That’s why the algorithm needs to pay less attention to the instances while constructing subsequent models.  
```
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_samples = 1000, n_features = 10,n_informative = 2, n_redundant = 0,random_state = 0, shuffle = False)
ADBclf = AdaBoostClassifier(n_estimators = 100, random_state = 0)
ADBclf.fit(X, y)
```
**Gradient Tree Boosting**  
It is also called Gradient Boosted Regression Trees (GRBT). It is basically a generalization of boosting to arbitrary differentiable loss functions. It produces a prediction model in the form of an ensemble of week prediction models. It can be used for the regression and classification problems. Their main advantage lies in the fact that they naturally handle the mixed type data.
```
from sklearn.ensemble import GradientBoostingClassifier
GDB_clf = GradientBoostingClassifier(n_estimators = 50, learning_rate = 1.0, max_depth = 1, random_state = 0)
GDB_clf.fit(X_train, y_train)
GDB_clf.score(X_test, y_test)
```

### 4. Dimentionality Reduction  
Dimensionality reduction, an unsupervised machine learning method is used to reduce the number of feature variables for each data sample selecting set of principal features. Principal Component Analysis (PCA) is one of the popular algorithms for dimensionality reduction.  

**Principal Component Analysis (PCA)**  
```
from sklearn.decomposition import PCA
Import scikitplot as skplt
pca = PCA(n_components=0.95)
pca_model = pca.fit_transform(X_train)
skplt.decomposition.plot_pca_2d_projection(pca, X, y)
```
**Kernel PCA**  
Kernel Principal Component Analysis, an extension of PCA, achieves non-linear dimensionality reduction using kernels. It supports both transform and inverse_transform.  
```
from sklearn.decomposition import KernelPCA
transformer = KernelPCA(n_components = 10, kernel = 'sigmoid')
X_transformed = transformer.fit_transform(X)
X_transformed.shape
```
**LDA**  
Linear Discriminant Analysis (LDA) is for supervised dimensionality reduction. It projects the input data to a linear subspace consisting of the directions which maximize the separation between classes, It is very useful in a multiclass setting.
```
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis()
clf.fit(X, y)
```

### 5. Clustering

**K Means**  
K-means clustering algorithm computes the centroids and iterates until we it finds optimal centroid. It assumes that the number of clusters are already known. The number of clusters identified from data by algorithm is represented by ‘K’ in K-means.  
```
from sklearn.cluster import KMeans
import scikitplot as skplt
k_means = KMeans(n_clusters=3, random_state=0)
skplt.cluster.plot_elbow_curve(kmeans, cluster_ranges=range(1, 30))
k_means.fit(X_train)
y_pred = k_means.predict(X_test)

```
**Mean-Shift**  
Mean_shift is another powerful clustering algorithm used in unsupervised learning. Mean-shift algorithm basically assigns the datapoints to the clusters iteratively by shifting points towards the highest density of datapoints i.e. cluster centroid. It is a non-parametric algorithm.  
```
from sklearn.cluster import MeanShift
ms = MeanShift()
ms.fit(X)
```
**Hierarchical Clustering**  
Hierarchical clustering is an unsupervised learning algorithm that is used to group together the unlabeled data points having similar characteristics. 
```
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')
cluster.fit_predict(X)
```



## Step 3: Evaluate  Model's Performance

### 1. Regression Metrics  
Regressor model performance: **Mean absolute error(MAE), Mean squared error(MSE), Median absolute error, Explain variance score, R2 score.**  
```
import sklearn.metrics as sm  
print("Regressor model performance:")
print("Mean absolute error(MAE) =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error(MSE) =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))
```

### 2. Classification Metrics  

**Accuracy Score**
```
>>> knn.score(X_test, y_test)
>>> from sklearn.metrics import accuracy_score
>>> accuracy_score(y_test, y_pred)
```
**Classification Report**
```
>>> from sklearn.metrics import classification_report
>>> print(classification_report(y_test, y_pred)))
```
**Confusion Matrix**
```
>>> from sklearn.metrics import confusion_matrix
>>> print(confusion_matrix(y_test, y_pred)))
```
**plot_confusion_matrix**
```
import scikitplot as skplt
skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
```

**plot_precision_recall**
```
import scikitplot as skplt
skplt.metrics.plot_precision_recall(y_test, y_probas)
```

**plot_roc**
```
import scikitplot as skplt
skplt.metrics.plot_roc(y_test, y_probas)
```

### 3. Clustering Metrics

**Adjusted Rand Index**
```
from sklearn.metrics import adjusted_rand_score
adjusted_rand_score(y_true, y_pred))
```
**Homogeneity**
```
from sklearn.metrics import homogeneity_score
homogeneity_score(y_true, y_pred))
```
**V-measure**
```
from sklearn.metrics import v_measure_score
metrics.v_measure_score(y_true, y_pred))
```
**plot_silhouette**
```
import scikitplot as skplt
skplt.metrics.plot_silhouette(X, kmeans.fit_predict(X))
```

### 4. Model validation  
**Cross-Validation**
```
import scikitplot as skplt
print(cross_val_score(knn, X_train, y_train, cv=4))
print(cross_val_score(lr, X, y, cv=2))
skplt.estimators.plot_learning_curve(model, X, y)
```



## Step 4:Tune Model


**Grid Search**  
Grid search is an approach to parameter tuning that will methodically build and evaluate a model for each combination of algorithm parameters specified in a grid.  
```
from sklearn.grid_search import GridSearchCV
params = {"n_neighbors": np.arange(1,3), "metric": ["euclidean", "cityblock"]}
grid = GridSearchCV(estimator=knn,param_grid=params)
grid.fit(X_train, y_train)
print(grid.best_score_)
print(grid.best_estimator_.n_neighbors)
```
**Randomized Parameter Optimization**   
Random search is an approach to parameter tuning that will sample algorithm parameters from a random distribution (i.e. uniform) for a fixed number of iterations. A model is constructed and evaluated for each combination of parameters chosen.  
```
from sklearn.grid_search import RandomizedSearchCV
params = {"n_neighbors": range(1,5), "weights": ["uniform", "distance"]}
rsearch = RandomizedSearchCV(estimator=knn,
   param_distributions=params,
   cv=4,
   n_iter=8,
   random_state=5)
rsearch.fit(X_train, y_train)
print(rsearch.best_score_)
```