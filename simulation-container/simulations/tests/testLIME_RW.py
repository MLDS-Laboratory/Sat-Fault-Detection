import lime
import lime.lime_tabular
import sklearn
from sklearn import ensemble
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import seaborn as sns
import shap
#weirdness occurred when i didn't have this
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_dir)
#import twoModeScenario as sc #this test is currently setup to run twoModeScenario, which only has pseudofaults
import simpleRWFaultScenario as sc
np.set_printoptions(threshold=np.inf)

#run the simulation, combine the data per timestep

iterate = True
SHAP = True

def runLIME():
    
        times, sigma, td, ta, faults, fric = sc.simulate(False)
        features = []
        times /= 1e9
        fric = np.array([(np.argmax(i) + 1, np.max(i)) for i in fric])
        for i in range(len(times)):
            #note that the time is included here but not passed into the classifier
            #it's only included here to make it easier to find the specific timestep being explained
            #since train_test_split changes the order

            #on a LIME-interested note, these values are absolute-valued because LIME's output was nonsensical otherwise. 
            #and that makes sense, because what's really the difference to it between a negative and positive torque?
            features.append([times[i]])
            features[i].extend(abs(td[i]))
            features[i].extend(abs(ta[:, i]))
            features[i].extend(fric[i])
        #features = [a + b + c for (a, b, c) in (sigma, torque_desired, torque_actual)]
        features = np.array(features)
        labels = [1 if i.any() else 0 for i in faults]
        #train and test sets, naming features and labels
        train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(features, labels, train_size=0.80)
        train = np.array(train)
        test = np.array(test)
        labels_train = np.array(labels_train)
        labels_test = np.array(labels_test)
        feature_names = ["td_1", "td_2", "td_3", "ta_1", "ta_2", "ta_3"]
        feature_names = np.array(feature_names)
        class_names = np.array(["No Fault", "Fault"])



        #training the classifier and getting results
        #note from here on out that the datasets passed in are indexed [1:] to exclude time
        rf = ensemble.RandomForestClassifier(n_estimators=500)
        rf.fit(train[:, 1:-2], labels_train)
        labels_pred = np.array(rf.predict(test[:, 1:-2]))
        #print(f"Random Forest Prediction Accuracy: \n{sklearn.metrics.classification_report(labels_test, labels_pred)}")
        #print(f"Accuracy = {rf.score(test[:, 1:-2], labels_test)}")


        #LIME!!!! passing all instances
        explainer = lime.lime_tabular.LimeTabularExplainer(train[:, 1:-2], 
                                                        feature_names=feature_names, 
                                                        class_names=class_names, 
                                                        discretize_continuous=False)


        #generating a random timestep for LIME to explain
        trues = np.array([i for i in range(len(labels_pred)) if labels_pred[i] == 1])#labels_pred[i] != ['NOMINAL'] * len(CSSdata[:, 0])]
        j = np.random.choice(trues, 1)[0]
        count = 0
        while np.argmax(rf.predict_proba(test[j-1:j+1, 1:-2])[1]) != labels_test[j]:
            j = np.random.choice(trues, 1)[0]
            if count > len(labels_test):
                print("NO CORRECT CLASSIFICATIONS")
                sys.exit()
        #CLASSIFIER MUST HAVE A PROBABILITY IN ITS PREDICTION
        #top_labels tells it how many labels to explain, in order from most to least likely
        fric_instance = test[j, -2:]
        exp = explainer.explain_instance(np.array(test[j, 1:-2]), rf.predict_proba, num_features=len(feature_names), top_labels=len(class_names))
        weights = sorted(exp.as_list(), key=lambda x: abs(x[1]), reverse=True)
        weights = np.array([int(i[0][-1]) for i in weights])
        
        indices = np.where(weights == int(fric_instance[0]))[0]
        quality.append((fric_instance[1], (indices[0], indices[1])))
        if SHAP:
            shap_explainer = shap.TreeExplainer(rf)
            shap_values = np.argsort(np.abs(shap_explainer.shap_values(test[j, 1:-2], labels_test[j])[:, 1]))[::-1]
            shap_values = np.array([i + 1 for i in shap_values])
            shap_indices = np.where(shap_values == int(fric_instance[0]))[0]
            shap_indices = np.concatenate((shap_indices, np.where(shap_values == int(fric_instance[0] + 3))[0]))
            shap_quality.append((fric_instance[1], (shap_indices[0], shap_indices[1])))
        else:
            shap_explainer = None
            
        

        return test, j, feature_names, labels_test, exp, trues, td, times, shap_explainer



if iterate:
    quality = []
    shap_quality = []
    num_error = 0
    count = 250
    for i in range(count):
        try:
            runLIME()
        except:
            num_error += 1
            continue
    
    mag = [i[0] for i in quality]
    weight = [1. / (i[1][0] + i[1][1]) for i in quality]
    sns.regplot(x=mag, y=weight, scatter=True, line_kws={'color':'red'})
    plt.title("Quality of LIME Explanation vs Friction Fault Magnitude")
    plt.ylabel("Derived Quality of Explanation")
    plt.xlabel("Maximum Magnitude of Viscous Friction Coefficient")
    plt.savefig("RW_Friction_Magnitude_LIME_Analysis_Derived_Quality_vs_Magnitude_5.png")

    print(f"ERROR PERCENTAGE: {num_error * 1. / count}")
    plt.figure()
    if len(shap_quality) > 1:
        mag = [i[0] for i in shap_quality]
        weight = [1. / (i[1][0] + i[1][1]) for i in shap_quality]
        sns.regplot(x=mag, y=weight, scatter=True, line_kws={'color':'red'})
        plt.title("Quality of SHAP Explanation vs Friction Fault Magnitude")
        plt.ylabel("Derived Quality of Explanation")
        plt.xlabel("Maximum Magnitude of Viscous Friction Coefficient")
        plt.savefig("RW_Friction_Magnitude_SHAP_Analysis_Derived_Quality_vs_Magnitude_3.png")
    plt.show()
else:
    test, j, feature_names, labels_test, rf, exp, trues, td, times, explainer = runLIME()
    #outputs
    print(f"Observation Explained:")
    print(f"time: {test[j][0]}")
    for i in range(len(feature_names)):
        print(f"{feature_names[i]}: {test[j, 1:-2][i]}")
    print(f"Actual Observation Label: {labels_test[j]}")
    print(f"Predicted Observation Label: {np.argmax(rf.predict_proba(test[j-1:j+1, 1:-2])[1])}")
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

    if SHAP:
        #SHAP STUFF
        plt.figure(3)
        
        shap_values = explainer.shap_values(test[:, 1:-2], labels_test)
        shap.waterfall_plot(shap.Explanation(values=shap_values[j, :, 1], base_values=explainer.expected_value[1], data=test[j, 1:-2], feature_names=feature_names))

        plt.figure(4)
        interaction = explainer.shap_interaction_values(test[j, 1:-2], labels_test[j])[:, :, 1]
        sns.heatmap(interaction, xticklabels=feature_names, yticklabels=feature_names, cmap='coolwarm', center=0)
        plt.title("SHAP Interaction Values for Explained Instance")

    plt.show()





