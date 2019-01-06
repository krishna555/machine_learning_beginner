from scipy.io import loadmat


mnist = loadmat("../datasets/classification/mnist-original.mat")
X = mnist["data"].T
y = mnist["label"][0]
#	X.shape
"""	(70000 , 784) => Each Image is constructed from 28x28 pixels. """
#	y.shape
"""	(70000, ) """

%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt 

some_digit = X[36000]

some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()

"""	Data well-distributed """ 

#	import collections
#	collections.Counter(y)

X_train, X_test, y_train, y_test = X[:60000] , X[60000:] , y[:60000], y[60000:]

import numpy as np 
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

"""			
		Binary Classifier
	The following code constructs a is5 Binary Classifier
"""

y_train_5 = (y_train == 5)
y_test_5 = (y_train == 5)

from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
is_digit_5 = sgd_clf.predict([some_digit])

""" O/P : True """

from sklearn.model_selection import cross_val_score

sgd_clf_cross_val = cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")

""" Excellent values for sgd_clf_cross_val observed ~95%. Weird :P """

from sklearn.base import BaseEstimator

class Never5Classifier(BaseEstimator):
	def fit(self, X, y=None):
		pass
	def predict(self, X, y=None):
		return [0]* len(X)

always_false_clf = Never5Classifier()

faulty_clf_cross_val = cross_val_score(always_false_clf, X_train, y_train_5, cv=3, scoring="accuracy")
"""
 Excellent values for always_false_clf observed. 
 Basically if the model always predicts isNot5, we are right 90% of the time
 Weird again... Let's not use accuracy as a metric for classification.
"""

from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)


from sklearn.metrics import (confusion_matrix, precision_score, recall_score, f1_score)
cf_matrix = confusion_matrix(y_train_5, y_train_pred)

sgd_precision = precision_score(y_train_5, y_train_pred)
sgd_recall = recall_score(y_train_5, y_train_pred)
sgd_f1 = f1_score(y_train_5, y_train_pred)

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
# print(y_scores)

from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

def plot_precision_recall_vs_thresholds(precisions, recalls, thresholds):
	plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
	plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
	plt.xlabel("Threshold")
	plt.legend(loc="upper left")
	plt.ylim([0, 1])

plot_precision_recall_vs_thresholds(precisions, recalls, thresholds)
plt.show()

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

def plot_roc_curve(fpr, tpr, label=None):
	plt.plot(fpr, tpr, linewidth=2, label=label)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.axis([0, 1, 0, 1])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')

plot_roc_curve(fpr, tpr)
plt.show()

from sklearn.metrics import roc_auc_score

print(roc_auc_score(y_train_5, y_scores))

""" Comparing ROC AUC With Random Forest Classifier """

from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier()
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")

y_scores_forest = y_probas_forest[:, 1]
fpr_forest, tpr_forest, thresholds_forest = precision_recall_curve(y_train_5, y_scores_forest)

plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right")
plt.show()

print(roc_auc_score(y_train_5, y_scores_forest))


""" 0.9928575682089069 Not bad at all """


""" 
	Multi Class Classification 
	One-versus-all Classification creates binary classifiers implicitly for all classes.
	Thus u would have 10 classifiers here for all 10 digits and the one with greatest decision score wins.
"""

sgd_clf.fit(X_train, y_train)
print(sgd_clf.predict([some_digit]))

some_digit_scores = sgd_clf.decision_function([some_digit])
print(some_digit_scores)
print(sgd_clf.classes_)

""" Random Forest Classifier """
forest_clf.fit(X_train, y_train)
print(forest_clf.predict([some_digit]))

print(forest_clf.predict_proba([some_digit]))


""" Let's improve SGD Classifier : StandardScaler to the rescue... """ 
print(cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy"))
""" ~85% """

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
""" >90% accuracy :)"""

