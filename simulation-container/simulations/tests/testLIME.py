import lime
import lime.lime_tabular
import sklearn
from sklearn import ensemble
import numpy as np
import matplotlib.pyplot as plt


size = 100
data = np.ones(int(size/10))
features = np.array([[x * i for x in data] for i in range(size)])
labels = np.array([1 if i > size/2 else 0 for i in range(size)])
train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(features, labels, train_size=0.80)

rf = ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(train, labels_train)

feature_names = np.array([i for i in range(size)])
class_names = np.array(["No Fault", "Fault"])

explainer = lime.lime_tabular.LimeTabularExplainer(train, 
                                                   feature_names=feature_names, 
                                                   class_names=class_names, 
                                                   discretize_continuous=True)
i = np.random.randint(0, test.shape[0])
exp = explainer.explain_instance(test[i], rf.predict_proba, num_features=int(size/10), top_labels=len(class_names))
print(test[i])
print(rf.predict_proba(test[i-1:i+1])[1])

fig = exp.as_pyplot_figure(exp.top_labels[0])
plt.show()






