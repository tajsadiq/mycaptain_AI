import pandas
from pandas.plotting import scatter_matrix
from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier

#Loading required datasets
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
names = ['sepal-length','sepal-width','petal-length','petal-width','class']
dataset = read_csv(url,names = names)
 
#Dataset dimension
print(dataset.shape)

#The Data
print(dataset.head(11))

#Statistical Data
print(dataset.describe())

#Class Distribution
print(dataset.groupby('class').size())

#UniVariate plots - box and whisker plots
dataset.plot(kind = 'box', subplots = True, layout = (2,2), sharex = False, sharey = False)
pyplot.show()

dataset.hist()
pyplot.show()

#Multivariate plots
scatter_matrix(dataset)
pyplot.show()

#Creating validation dataset
#splitting dataset
array = dataset.values
X = array[:, 0:4]
y = array[:, 4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size = 0.2, random_state = 1)

#Logistic Regression
#Linear Discriminant Analysis
#K-Nearest Neighbors
#Classification and Regression Treees
#Gaussian naive Bayes
#Support Vectir Machines

#building the models
models = []
models.append(('LR',LogisticRegression(solver = 'liblinear', multi_class = 'ovr')))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM', SVC(gamma = 'auto')))

#Evaluation
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits = 10)
    cv_results = cross_val_score(model, X_train, Y_train, cv = kfold, scoring = 'accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

#Comparison
pyplot.boxplot(results, labels = names)
pyplot.title('Algorithm Comparison')
pyplot.show()

#SVM Predictions
model = SVC(gamma = 'auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

#Prediction Evaluation
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
