import lime
import lime.lime_tabular
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
import simpleRWFaultScenario as sc
np.set_printoptions(threshold=np.inf)

#run the simulation, combine the data per timestep
times, sigma, td, ta, faults = sc.simulate(False)
features = []
times /= 1e9
for i in range(len(times)):
    #note that the time is included here but not passed into the classifier
    #it's only included here to make it easier to find the specific timestep being explained
    #since train_test_split changes the order

    #on a LIME-interested note, these values are absolute-valued because LIME's output was nonsensical otherwise. 
    #and that makes sense, because what's really the difference to it between a negative and positive torque?
    features.append([times[i]])
    features[i].extend(abs(td[i]))
    features[i].extend(abs(ta[:, i]))
#features = [a + b + c for (a, b, c) in (sigma, torque_desired, torque_actual)]
features = np.array(features)
labels = [1 if i.any() else 0 for i in faults]
#train and test sets, naming features and labels
train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(features, labels, train_size=0.80)
train = np.array(train)
test = np.array(test)
labels_train = np.array(labels_train)
labels_test = np.array(labels_test)
feature_names = ["td_x", "td_y", "td_z", "ta_x", "ta_y", "ta_z"]
feature_names = np.array(feature_names)
class_names = np.array(["No Fault", "Fault"])



#training the classifier and getting results
#note from here on out that the datasets passed in are indexed [1:] to exclude time
rf = ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(train[:, 1:], labels_train)
labels_pred = np.array(rf.predict(test[:, 1:]))
print(f"Random Forest Prediction Accuracy: \n{sklearn.metrics.classification_report(labels_test, labels_pred)}")
print(f"Accuracy = {rf.score(test[:, 1:], labels_test)}")


#LIME!!!! passing all instances
explainer = lime.lime_tabular.LimeTabularExplainer(train[:, 1:], 
                                                   feature_names=feature_names, 
                                                   class_names=class_names, 
                                                   discretize_continuous=False)


#generating a random timestep for LIME to explain
trues = np.array([i for i in range(len(labels_pred)) if labels_pred[i] == 1])#labels_pred[i] != ['NOMINAL'] * len(CSSdata[:, 0])]
j = np.random.choice(trues, 1)[0]
count = 0
while np.argmax(rf.predict_proba(test[j-1:j+1, 1:])[1]) != labels_test[j]:
    j = np.random.choice(trues, 1)[0]
    if count > len(labels_test):
        print("NO CORRECT CLASSIFICATIONS")
        sys.exit()
#CLASSIFIER MUST HAVE A PROBABILITY IN ITS PREDICTION
#top_labels tells it how many labels to explain, in order from most to least likely
exp = explainer.explain_instance(np.array(test[j, 1:]), rf.predict_proba, num_features=len(feature_names), top_labels=len(class_names))

#outputs
print(f"Observation Explained:")
print(f"time: {test[j][0]}")
for i in range(len(feature_names)):
    print(f"{feature_names[i]}: {test[j, 1:][i]}")
print(f"Actual Observation Label: {labels_test[j]}")
print(f"Predicted Observation Label: {np.argmax(rf.predict_proba(test[j-1:j+1, 1:])[1])}")
print(f"LIME model bias for predicted class: {exp.intercept[0]}")
print(f"LIME model bias for other class: {exp.intercept[1]}")
print(f"R^2 of LIME's local linear model: {exp.score}")

plt.figure(1)
#mrpFeedback Desired Torque Outputs
for i in range(len(td[0])):
    plt.plot(times, td[:, i], label=f'RW {i+1}')
true_preds = np.array(test[trues])
actual_preds = np.array([test[i][0] for i in range(len(labels_test)) if labels_test[i] == 1])
for i in actual_preds:
    plt.axvline(x=i, color='g')
for i in true_preds[:, 0]:
    plt.axvline(x=i, linestyle='--', color='r')
plt.title("mrpFeedback Desired Torques")
plt.legend()
plt.xlabel("Time [orbits]")
plt.ylabel("Torque [N-m]")
plt.xlim(test[j][0] - 0.1, test[j][0] + 0.1)

fig = exp.as_pyplot_figure(exp.top_labels[0])
plt.show()
