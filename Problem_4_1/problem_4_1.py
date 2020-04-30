import matplotlib.pyplot as plt
import numpy as np
import time

from sklearn.datasets import fetch_openml
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle


def performance(model, X_train, y_train, X_test, y_test):
	start = time.time()
	model.fit(X_train, y_train)
	train = time.time()
	predictions = model.predict(X_test)
	stop = time.time()
	train_errs = model.score(X_train, y_train)
	test_errs = model.score(X_test, y_test)
	print("Train time" + str(train-start))
	print("Train errors: " + str(train_errs))
	print("Test errors: " + str(test_errs))
	accuracy = accuracy_score(y_test, predictions)
	train_time = train-start
	test_time = stop - train
	return accuracy, test_time

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

#take subset of data so it doesn't take forever 
X_shuffle, y_shuffle = shuffle(X, y, random_state=0)
X_trim = X_shuffle[0:40000]
y_trim = y_shuffle[0:40000]

X_train, X_test, y_train, y_test = train_test_split( X_trim, y_trim, random_state=42)


# KNN classifier rewrote mnistknndemo.m from HW1
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn_accuracy, knn_time = performance(knn, X_train, y_train, X_test, y_test)
print('Accuracy for KNN: ' + str(knn_accuracy * 100))
print('Total execution time for KNN: ' + str(knn_time))

#multi-class logistic regression from HW1 
logi_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
logi_accuracy, logi_time = performance(logi_reg, X_train, y_train, X_test, y_test)
print('Accuracy for Logistic Regression: ' + str(logi_accuracy * 100))
print('Total execution time for Logistic Regression: ' + str(logi_time))

#SVM from 3.2
svm_model = SVC(C=3.4, decision_function_shape='ovr', gamma="scale", kernel='rbf')
svm_accuracy, svm_time = performance(svm_model, X_train, y_train, X_test, y_test)
print('Accuracy for SVM: ' + str(svm_accuracy * 100))
print('Total execution time for SVM Classifier: ' + str(svm_time))

decision_tree_model = DecisionTreeClassifier()
decision_accuracy, deicison_time = performance(decision_tree_model, X_train, y_train, X_test, y_test)
print('Accuracy for Decision Tree Classifier: ' + str(decision_accuracy * 100))
print('Total execution time for Decision Tree Classifier: ' + str(deicison_time))


bagging_classifier_model = BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5, max_features=0.5)
bagging_accuracy, bagging_time = performance(bagging_classifier_model, X_train, y_train, X_test, y_test)
print('Accuracy for Bagging  Classifier: ' + str(bagging_accuracy * 100))
print('Total execution time for Bagging Classifier: ' + str(bagging_time))


random_forest_model = RandomForestClassifier(max_depth=10, random_state=0)
forest_accuracy, forest_time = performance(random_forest_model, X_train, y_train, X_test, y_test)
print('Accuracy for Random Forest Classifier: ' + str(forest_accuracy * 100))
print('Total execution time for Random Forest  Classifier: ' + str(forest_time))


estimators = [('rf', RandomForestClassifier(n_estimators=10, random_state=42)), ('knn', KNeighborsClassifier(n_neighbors=5))]
stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000))
stacking_accuracy, stacking_time = performance(stacking_model, X_train, y_train, X_test, y_test)
print('Accuracy for Stacking Model Classifier: ' + str(stacking_accuracy * 100))
print('Total execution time for Stacking Model Classifier: ' + str(stacking_time))