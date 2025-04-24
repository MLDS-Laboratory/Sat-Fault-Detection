import shap
import sklearn
from sklearn import ensemble
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
#weirdness occurred when i didn't have this
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_dir)
#import twoModeScenario as sc #this test is currently setup to run twoModeScenario, which only has pseudofaults
import simpleCSSFaultScenario as sc
np.set_printoptions(threshold=np.inf)

#run the simulation, combine the data per timestep
times, sigma, sensedSun, CSSdata, faults = sc.simulate(False, False)
features = []
times /= 1e9
for i in range(len(times)):
    #note that the time is included here but not passed into the classifier
    #it's only included here to make it easier to find the specific timestep being explained
    #since train_test_split changes the order

    #on a LIME-interested note, these values are absolute-valued because LIME's output was nonsensical otherwise. 
    #and that makes sense, because what's really the difference to it between a negative and positive torque?
    features.append([times[i]])
    #features[i].extend(abs(sigma[i]))
    #features[i].extend(abs(sensedSun[i]))
    features[i].extend(CSSdata[:, i])
#features = [a + b + c for (a, b, c) in (sigma, torque_desired, torque_actual)]
features = np.array(features)
class_names = np.array(["OFF", "STUCK_CURRENT", "STUCK_MAX", "STUCK_RAND", "RAND", "NOMINAL"])
labels = np.array([1 if np.isin(i, class_names[:-1]).any() else 0 for i in faults])

#train and test sets, naming features and labels
train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(features, labels, train_size=0.80)
train = np.array(train)
test = np.array(test)
labels_train = np.array(labels_train)
labels_test = np.array(labels_test)
feature_names = []#["sigma_x", "sigma_y", "sigma_z"]
CSS_names = [f"CSS_{i+1}" for i in range(len(CSSdata[:, 0]))]
feature_names.extend(CSS_names)
feature_names = np.array(feature_names)
class_names = np.array(["No Fault", "Fault"])



#training the classifier and getting results
#note from here on out that the datasets passed in are indexed [1:] to exclude time
rf = ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(train[:, 1:], labels_train)
labels_pred = np.array(rf.predict(test[:, 1:]))
print(f"Random Forest Prediction Accuracy: \n{sklearn.metrics.classification_report(labels_test, labels_pred)}")
print(f"Accuracy = {rf.score(test[:, 1:], labels_test)}")


plt.figure(1, figsize=(20,len(CSSdata[:, 0]) / 2))
colors = plt.cm.tab20.colors[:len(CSSdata[:, 0])]
for i in range(len(CSSdata[:, 0])):
    plt.plot(times, faults[:, i], label=rf"$CSS_{{{i+1}}}$", color=colors[i])
plt.title("CSS Sensors' Fault State")
plt.xlabel("Time [orbits]")
plt.ylabel("Fault State")
plt.legend()

plt.figure()

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(test[:, 1:], labels_test)
shap.summary_plot(shap_values[:, :, 1], test[:, 1:], feature_names=feature_names)

plt.show()