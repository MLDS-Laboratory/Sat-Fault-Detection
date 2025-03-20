import lime
import lime.lime_tabular
import sklearn
from sklearn import ensemble
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_dir)
import simpleRWFaultScenario as sc
times, sigma, torque_desired, torque_actual, faults = sc.simulate(False)
torque_actual = np.array(torque_actual)
times = times / np.pow(10, 9)
features = []
for i in range(len(times)):
    features.append([])
    features[i].extend(sigma[i])
    features[i].extend(torque_desired[i])
    features[i].extend(torque_actual[:, i])
#features = [a + b + c for (a, b, c) in (sigma, torque_desired, torque_actual)]
features = np.array(features)
labels = np.max(faults, axis=1)

#size = 100
#data = np.ones(int(size/10))
#features = np.array([[x * i for x in data] for i in range(size)])
#labels = np.array([1 if i > size/2 else 0 for i in range(size)])
train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(features, labels, train_size=0.80)

rf = ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(train, labels_train)
print(f"Random Forest Prediction Accuracy: {sklearn.metrics.accuracy_score(labels_test, rf.predict(test))}")

#feature_names = np.array([i for i in range(size)])
feature_names = np.array(["sigma_x", "sigma_y", "sigma_z", "td_1", "td_2", "td_3", "ta_1", "ta_2", "ta_3"])
class_names = np.array(["No Fault", "Fault"])

explainer = lime.lime_tabular.LimeTabularExplainer(train, 
                                                   feature_names=feature_names, 
                                                   class_names=class_names, 
                                                   discretize_continuous=False)
i = np.random.randint(0, test.shape[0])
exp = explainer.explain_instance(test[i], rf.predict_proba, num_features=len(feature_names), top_labels=len(class_names))
print(f"Observation Explained: {test[i]}")
print(f"Actual Observation Label: {labels_test[i]}")
print(f"Predicted Observation Label: {np.argmax(rf.predict_proba(test[i-1:i+1])[1])}")

fig = exp.as_pyplot_figure(exp.top_labels[0])
plt.show()






