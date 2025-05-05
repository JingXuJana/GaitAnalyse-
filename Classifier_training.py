import numpy as np
import os
import torch
import random
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import KFold, train_test_split, LeaveOneGroupOut
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, roc_curve, auc
from itertools import combinations
import torch
from scipy.interpolate import interp1d

# Example data segment (replace with your actual data)
#   # replace with your actual gait data
# original_time = np.linspace(0, 0.35, 70)  # for 70 data points at 200 Hz

# # New time axis for 80 points
# new_time = np.linspace(0, 0.35, 80)

# # Linear interpolation
# interpolate = interp1d(original_time, original_data, kind='linear')
# resampled_data = interpolate(new_time)



def interpolate_list(lst):
    original_indices = np.linspace(0, len(lst)-1, num=len(lst))
    new_indices = np.linspace(0, len(lst)-1, num=160)
    return np.interp(new_indices, original_indices, lst)


def ground_truth(datafile):
    new_list = torch.load(datafile)
    long_list = []
    for i, j in zip(new_list , range(len(new_list))):
        flat_list = []
        for lst in i:
            new_lst = interpolate_list(lst)
            flat_list.extend(new_lst)
        long_list.append(flat_list)
    return long_list
    




def load_weights_and_labels(dict_path, diag_covar):
    # from the result dictionary of the MP Algorithm get weight mean and weight covariance,
    # put the mean and weight of one segment together as one flatt list. 

    results = torch.load(dict_path)
    MP_results = results["Weights"]
   # MP_results=  dict(list(MP_results.items())[:28])
    segment_weight_list = []
    label_list = []
    subject_list = []

    for i, path in enumerate(MP_results):
        Segment_Weighs = MP_results[path]
        for segment in Segment_Weighs:
            subject_list.append(i+1)
            if "MPs" in results and i <= 14: # the first 14 paths are norm data 
                label_list.append(0)
            elif "MPs" in results and i > 14:
                label_list.append(1)
            elif "MPs" not in results and i <= 5: # the first 5 paths are norm data 
                label_list.append(0) 
            elif "MPs" not in results and i > 5:
                label_list.append(1)

            weight_mean = Segment_Weighs[segment]['Weight_mean']
            weight_cov = Segment_Weighs[segment]['Weight_covariance'] 
            mean_vector = weight_mean.view(-1)
            cov_vector = weight_cov.view(42, -1)

            if diag_covar: 
                weight_cov_diag = [torch.diag(matrix) for matrix in weight_cov]
                cov_diag_vector = torch.cat(weight_cov_diag,dim = 0)
                weights_for_one = torch.cat((mean_vector, cov_diag_vector), dim=0).numpy()

            if not diag_covar: 
                weights_for_one = torch.cat((mean_vector, cov_vector.view(-1)), dim=0).numpy()
            
            segment_weight_list.append(weights_for_one)
    
    print (f"I am here checking the feature size, so... it is {len(weights_for_one)}")
     # this is the flatt weight list of all the participants together. 
    return label_list, segment_weight_list, subject_list


class classification:
    def __init__(self, weight_list, label_list,subject_list,classifier,test_weight_list,test_label_list,test_subject_list,random_state):
        # the weight matrix, from load_weights_and_labels function. 
        self.weight_list = weight_list # features matrix 
        self.label_list = np.array(label_list) # list of 1s and 0s 
        self.subjects =  np.array(subject_list) # (Your array of subject identifiers, each unique to a particular subject)
        # ask to make sure, if I need to scale the features so that they have a mean of 0 and a variance of 1.
        scaler = StandardScaler()
        self.random_state = random_state
        self.weight_scaled = scaler.fit_transform(self.weight_list)
        self.test_weight_list = test_weight_list
        self.test_weight_scaled = scaler.fit_transform(self.test_weight_list)
        self.test_label_list = test_label_list
        self.test_subject_list = test_subject_list
        self.classifier = classifier

    def models_grifSearch(self): # define classification model and parameter grip for the CV 
        # using SVM with linear kernel
        if self.classifier == "SVM_linear":
            self.model = SVC(kernel='linear', probability=True)
            self.param_grid = {
            'C': [0.001]}
        # using SVM with rbf kernel
        if self.classifier == "SVM_rbf":
            self.model = SVC(kernel='rbf',probability=True)
            self.param_grid = {
            'C': [10, 100], 
            'gamma': ['scale', 10**-5]}
        # using random forest 
        if self.classifier == "RF":
            self.model = RandomForestClassifier(random_state=self.random_state)
            # extremely long running time for parameter tuning especially by LOGO
            self.param_grid = {
            # 'max_depth': [20, 60, 100, None], # mostly None 
            'max_features': ['sqrt',],
            'n_estimators': [200, 400, 600, 800, 1600,3200,6400]}  # it seems the more the better ,800,1600,3200,6400,12800
            
            # self.param_grid = {'bootstrap': [True, False],
            # 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
            # 'max_features': ['auto', 'sqrt'],
            # 'min_samples_leaf': [1, 2, 4],  
            # 'min_samples_split': [2, 5, 10],
            # 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
        # kNeigbors 
        if self.classifier == "KNN":
            self.model = KNeighborsClassifier()
            self.param_grid = {
            'n_neighbors': (1,2,3,4,5,6,7,8,9,10,12,13),  
             'leaf_size': (5,10,15,20),
            'p': (1,2),
            'metric': ('minkowski', 'chebyshev')

            # 'n_neighbors': (1,10, 3),
            # 'leaf_size': (20,40,5),
            # 'p': (1,2),

           # 'metric': ('minkowski', 'chebyshev')
            }
        return self.model, self.param_grid


    def ten_fold_CV(self):

        # 1/10 of testing data, 9/10 of training data 
        X_train = self.weight_scaled
        y_train = self.label_list
        X_test = self.test_weight_scaled
        self.y_test = self.test_label_list
        k_fold = KFold(n_splits=10, shuffle=True, random_state = self.random_state)
        self.model,self.param_grid = self.models_grifSearch()

        self.grid_search_cv = GridSearchCV(self.model, self.param_grid, cv=k_fold, scoring='accuracy')
        # fit the grid search with the training data
        self.grid_search_cv.fit(X_train, y_train)
        self.best_params = self.grid_search_cv.best_params_
        self.training_score = self.grid_search_cv.best_score_

        print("Best parameters of the {} after gridSearch with 10-fold validation are:\n{}".format(self.classifier,self.grid_search_cv.best_params_))
        print("Best score is: {} " .format(self.grid_search_cv.best_score_))
		# Evaluate the best parameters on the test set
        y_pred = self.grid_search_cv.predict(X_test)
        self.test_score = self.grid_search_cv.score(X_test, self.y_test)
        self.conf_matrix = confusion_matrix(self.y_test, y_pred)
       
        print(f"Confusion Matrix:{self.conf_matrix}")
        print(f"Accuracy on the test set with the best parameters is: {self.test_score} ")
        self. y_probs = self.grid_search_cv.predict_proba(X_test)[:, 1]  # Assuming the positive class is at index 1

        return self. training_score, self.classifier, self.test_score, self.best_params , self.conf_matrix

        # result = permutation_importance(grid_search_cv, X_train, y_train, n_repeats=10, random_state=42)

# Get indices of top features based on importance scores
        # top_feature_indices = np.argsort(result.importances_mean)[::-1][:10]
        # print (f"The most important features are {top_feature_indices}")


        # filename = f"results_10Fold_{self.classifier}.txt"

        # with open(filename, "w") as f:
        #     f.write(f"Classifier:{self.classifier}\nValidation:10 fold Validation\n\n")
        #     f.write(f"Accuracy of the 10 fold Validation on the test set is {score}\n\n")
        #     f.write(f"The numpy random state is {current_random_state}")

    def leave_one_group_out(self):

        logo = LeaveOneGroupOut()
        X_train = self.weight_scaled
        y_train = self.label_list
        X_test = self.test_weight_scaled
        self.y_test = self.test_label_list
        groups_train = self.subjects
        group_test = self.test_subject_list
        self.model,self.param_grid = self.models_grifSearch()


        grid_search = GridSearchCV(self.model,
                                    param_grid=self.param_grid,
                                    cv=logo.split(X_train, y_train, groups=groups_train),
                                    scoring='accuracy')
        grid_search.fit(X_train, y_train, groups=groups_train)
        self.training_score = grid_search.best_score_
        print(self. training_score)
        
        print(grid_search.best_params_)
        self.best_estimator = grid_search.best_estimator_
        print(grid_search.best_estimator_)

        self.best_params = grid_search.best_params_
        y_pred = self.best_estimator.predict(X_test)
        self.test_score = grid_search.score(X_test, self.y_test)
        self.conf_matrix = confusion_matrix(self.y_test, y_pred)

        self.y_probs = grid_search.predict_proba(X_test)[:, 1]  # Assuming the positive class is at index 1


        if self.classifier == "SVM_linear":
            best_svm = self.best_estimator
            importances = np.abs(best_svm.coef_[0])
            top_feature_indices = np.argsort(importances)[::-1][:10]
            print (f"The most important features are {top_feature_indices }")


        print(f"The accuracy of the {self.classifier} classifier is {self.test_score}")

        return self.training_score, self.classifier, self.test_score, self.best_params, self.conf_matrix


    def ROC_curve(self,validation):
        if validation == "tenfold":
            self.ten_fold_CV()
        if validation == "LOGO":
            self.leave_one_group_out()
        
        fpr, tpr, _ = roc_curve(self.y_test, self.y_probs)
        roc_auc = auc(fpr, tpr)
        print(f"Confusion Matrix:{self.conf_matrix}")
        print(f"Accuracy on the test set with the best parameters is: {self.test_score} ")


        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

    def PCA_component(self):
        self.pca = PCA(n_components=5)
        self.X_pca = self.pca.fit_transform(self.weight_scaled)
        # self.X_pca_train = pca.fit_transform(self.weight_scaled)
        self.X_pca_test = self.pca.transform(self.test_weight_scaled)

   
    def plot_decision_boundaries(self,X_pca, X_pca_test, clf, ax, pc1, pc2):
        self.PCA_component()
        x_min, x_max = X_pca[:, pc1].min() - 1, X_pca[:, pc1].max() + 1
        y_min, y_max = X_pca[:, pc2].min() - 1, X_pca[:, pc2].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 120), np.linspace(y_min, y_max, 120))

        other_components = np.zeros((xx.size, self.pca.n_components_ - 2))
        pca_space_grid = np.column_stack((xx.ravel(), yy.ravel(), other_components))


        pca_space_grid[:, pc1] = xx.ravel()
        pca_space_grid[:, pc2] = yy.ravel ()

        original_space_grid = self.pca.inverse_transform(pca_space_grid)

        # Predict using the classifier for each point in the mesh grid
        Z = clf.predict(original_space_grid)
        Z = Z.reshape(xx.shape)

    # Plot the contour and training points
        ax.contourf(xx, yy, Z, alpha=0.3)


        # Define colors for each class
        train_colors = ['goldenrod', 'steelblue']  # Colors for training classes
        test_colors = ['darkorange', 'dodgerblue']   # Colors for test classes

        # Plot data points by class
        for class_value in [0, 1]:
            # Plot training data for this class
            ix_train = np.where(self.label_list == class_value)
            if class_value == 0:
                indi = "HC"
            if class_value == 1:
                indi = "Pat"

            ax.scatter(X_pca[ix_train, pc1], X_pca[ix_train, pc2], label=f'Train {indi}', color=train_colors[class_value], alpha = 0.5, s=10)

            # Plot test data for this class
            ix_test = np.where(np.array(self.test_label_list) == class_value)
            ax.scatter(X_pca_test[ix_test, pc1], X_pca_test[ix_test, pc2], marker='x', alpha=1, label=f'Test {indi}', color=test_colors[class_value], s=20)

        ax.set_xlabel(f'PC{pc1+1}')
       # ax.set_ylabel(f'PC{pc2+1}')
        ax.legend(fontsize='large')
       # ax.title(f"PC{pc1+1}/PC{pc2+1}")

    def visualization_PCA(self):
        self.PCA_component()
        # Plotting the pairwise scatter plots with decision boundaries
        # fig, axes = plt.subplots(5, 5, figsize=(20, 20))

# Plot each combination of principal components
        # for i, j in combinations(range(5), 2):
        #     self.plot_decision_boundaries(self.X_pca, self.X_pca_test, self.best_estimator,axes[i, j], i, j) #  self.best_estimator, self.grid_search_cv.best_estimator_, 

        # # Turn off the plots for repeated combinations or i==j
        # for i in range(5):
        #     for j in range(5):
        #         if i >= j:
        #             axes[i, j].axis('off')

        # plt.tight_layout()
        # plt.show()

        
        

    # Plot each combination of principal components separately
        
        for i, j in combinations(range(5), 2):
            plt.figure(figsize=(7, 5))  # Create a new figure for each plot
           # self.plot_decision_boundaries(self.X_pca, self.X_pca_test, self.grid_search_cv.best_estimator_, plt.gca(), i, j)
            self.plot_decision_boundaries(self.X_pca, self.X_pca_test,  self.best_estimator,  plt.gca(), i, j)


# Turn off the plots for repeated combinations or i==j
        # for i in range(5):
        #     for j in range(5):
        #         if i >= j:
        #             axes[i, j].axis('off')

             
            plt.title(f'PCA Components {i} vs {j}')
            plt.tight_layout()
            plt.show()



    
    
        # if self.classifier == "SVM_linear":
        #     best_svm = grid_search.best_estimator_
        #     importances = np.abs(best_svm.coef_[0])
        #     top_feature_indices = np.argsort(importances)[::-1][:10]
        #     print (f"The most important features are {top_feature_indices}")

if __name__ == "__main__":
    random_numbers = [random.randint(1, 1000) for _ in range(5)]
    datafile = "/Users/janax/mp_binning/MP_training_data_list.pth"
    label_list = ground_truth(datafile)

    SVM_linear_training_score_list_tenfold = []
    SVM_linear_test_score_list_tenfold = []
    SVM_linear_matrix_tenfold = []

    SVM_linear_training_score_list_LOGO = []
    SVM_linear_test_score_list_LOGO = []
    SVM_linear_matrix_LOGO = []

    SVM_RBF_training_score_list_tenfold = []
    SVM_RBF_test_score_list_tenfold = []
    SVM_RBF_matrix_tenfold = []

    SVM_RBF_training_score_list_LOGO = []
    SVM_RBF_test_score_list_LOGO = []
    SVM_RBF_matrix_LOGO = []

    kNN_training_score_list_tenfold = []
    kNN_test_score_list_tenfold = []
    kNN_matrix_tenfold = []

    kNN_training_score_list_LOGO = []
    kNN_test_score_list_LOGO = []
    kNN_matrix_LOGO = []

    RF_training_score_list_tenfold = []
    RF_test_score_list_tenfold = []
    RF_matrix_tenfold = []

    RF_training_score_list_LOGO = []
    RF_test_score_list_LOGO = []
    RF_matrix_LOGO = []


    for diag_covar in [False]:
        dict_path = os.path.join('weight_dict','results_for_all_MP16.pkl')
        label_list,oldsegment_weight_list,subject_list  = load_weights_and_labels(dict_path=dict_path, diag_covar = diag_covar)
        test_weight_path = os.path.join('weight_dict','results_weight_16.pkl')
        test_label_list,oldtest_weight_list,test_subject_list = load_weights_and_labels(dict_path=test_weight_path,diag_covar = diag_covar)
        
        datafile = "/Users/janax/mp_binning/MP_training_data_list.pth"
        segment_weight_list = ground_truth(datafile)
        test_data_file = "/Users/janax/mp_binning/Test_data_list.pth"
        test_weight_list = ground_truth(test_data_file)
        
        if diag_covar:
            filename = "results_diag_covar_SVM.txt"
            picklename = "diag_covar_SVM.pkl"
        else:
            filename = "results_groud_Truth.txt"
            picklename = "ground_Truth_5.pkl"
        



        for random_state in random_numbers:
            with open(filename, "a") as f:
                f.write (f"The random state is {random_state}\n\n")


            # classifier_list = ["SVM_linear", "SVM_rbf","KNN","RF"]
            # classifiers = {}

            # for model in classifier_list: 
            #     classifiers[model] = classification(weight_list=segment_weight_list, label_list=label_list, subject_list=subject_list, classifier=model,
            #                                      test_weight_list=test_weight_list, test_label_list=test_label_list, test_subject_list=test_subject_list, random_state=random_state)

            #     training_score, classifier, test_score, best_params, conf_matrix = classifiers[model].ten_fold_CV()

            #     with open(filename, "a") as f:
            #         f.write("Best parameters of the {} after gridSearch with 10-fold validation are:\n{}".format(classifier,best_params))
            #         f.write(f" The training score is {training_score}\n")
            #         f.write(f"The confusion matrix is: {conf_matrix}")
            #         f.write(f"Accuracy of the 10 fold Validation on the test set is {test_score}\n\n\n")
                            
            #     SVM_linear_training_score_list_tenfold.append (training_score)
            #     SVM_linear_test_score_list_tenfold.append(test_score)
            #     SVM_linear_matrix_tenfold.append(conf_matrix)

            #     training_score,classifier, test_score, best_params, conf_matrix = classifiers[model].leave_one_group_out()
            #     kNN_training_score_list_LOGO = training_score
            #     kNN_test_score_list_LOGO = test_score
            #     kNN_matrix_LOGO = conf_matrix 
            #     with open(filename, "a") as f:
            #         f.write("Best parameters of the {} after gridSearch with LOGO validation are:\n{}".format(classifier,best_params))
            #         f.write(f" The training score is {training_score}\n")
            #         f.write(f"The confusion matrix is: {conf_matrix}")
            #         f.write(f"Accuracy of the 10 fold Validation on the test set is {test_score}\n\n\n")



            SVM_linear = classification(weight_list=segment_weight_list,label_list=label_list,subject_list=subject_list,classifier="SVM_linear",
                                        test_weight_list=test_weight_list,test_label_list=test_label_list,test_subject_list = test_subject_list,random_state=random_state)


            # training_score, classifier, test_score, best_params, conf_matrix = SVM_linear.ten_fold_CV()
        
            # with open(filename, "a") as f:
            #     f.write("Best parameters of the {} after gridSearch with 10-fold validation are:\n{}".format(classifier,best_params))
            #     f.write(f" The training score is {training_score}\n")
            #     f.write(f"The confusion matrix is: {conf_matrix}")
            #     f.write(f"Accuracy of the 10 fold Validation on the test set is {test_score}\n\n\n")
            
            # SVM_linear.visualization_PCA()
                
            # SVM_linear_training_score_list_tenfold.append (training_score)
            # SVM_linear_test_score_list_tenfold.append(test_score)
            # SVM_linear_matrix_tenfold.append(conf_matrix)


            SVM_rbf = classification(weight_list=segment_weight_list,label_list=label_list,subject_list=subject_list,classifier="SVM_rbf",
                                        test_weight_list=test_weight_list,test_label_list=test_label_list,test_subject_list = test_subject_list, random_state=random_state)
            
            # training_score,classifier, test_score, best_params, conf_matrix = SVM_rbf.ten_fold_CV()
            # SVM_RBF_training_score_list_tenfold.append (training_score)
            # SVM_RBF_test_score_list_tenfold .append(test_score)
            # SVM_RBF_matrix_tenfold .append(conf_matrix)
            # with open(filename, "a") as f:
            #     f.write("Best parameters of the {} after gridSearch with LOGO validation are:\n{}".format(classifier,best_params))
            #     f.write(f" The training score is {training_score}\n")
            #     f.write(f"The confusion matrix is: {conf_matrix}")
            #     f.write(f"Accuracy of the 10 fold Validation on the test set is {test_score}\n\n\n")

            #  SVM_rbf.visualization_PCA()




            KNeighbors = classification(weight_list=segment_weight_list,label_list=label_list,subject_list=subject_list,classifier="KNN",
                                        test_weight_list=test_weight_list,test_label_list=test_label_list,test_subject_list = test_subject_list,random_state=random_state)
            
            # training_score,classifier, test_score, best_params, conf_matrix = KNeighbors.ten_fold_CV()
            # # kNN_training_score_list_tenfold.append (training_score)
            # # kNN_test_score_list_tenfold .append(test_score)
            # # kNN_matrix_tenfold .append(conf_matrix)
            # # with open(filename, "a") as f:
            # #     f.write("Best parameters of the {} after gridSearch with LOGO validation are:\n{}".format(classifier,best_params))
            # #     f.write(f" The training score is {training_score}\n")
            # #     f.write(f"The confusion matrix is: {conf_matrix}")
            # #     f.write(f"Accuracy of the 10 fold Validation on the test set is {test_score}\n\n\n")
        
            # KNeighbors.visualization_PCA()


            random_forest = classification(weight_list=segment_weight_list,label_list=label_list,subject_list=subject_list,classifier="RF",
                                        test_weight_list=test_weight_list,test_label_list=test_label_list,test_subject_list = test_subject_list,random_state=random_state)

            # training_score,classifier, test_score, best_params, conf_matrix=random_forest.ten_fold_CV()
            # RF_training_score_list_tenfold. append (training_score)
            # RF_test_score_list_tenfold. append(test_score)
            # RF_matrix_tenfold. append(conf_matrix)
            # with open(filename, "a") as f:
            #     f.write("Best parameters of the {} after gridSearch with LOGO validation are:\n{}".format(classifier,best_params))
            #     f.write(f" The training score is {training_score}\n")
            #     f.write(f"The confusion matrix is: {conf_matrix}")
            #     f.write(f"Accuracy of the 10 fold Validation on the test set is {test_score}\n\n\n")
            # random_forest.ROC_curve("tenfold")

            # random_forest.visualization_PCA()


            training_score, classifier, test_score, best_params, conf_matrix = SVM_linear.leave_one_group_out()
            with open(filename, "a") as f:
                f.write("Best parameters of the {} after gridSearch with LOGO validation are:\n{}".format(classifier,best_params))
                f.write(f" The training score is {training_score}\n")
                f.write(f"The confusion matrix is: {conf_matrix}")
                f.write(f"Accuracy of the 10 fold Validation on the test set is {test_score}\n\n\n")
            SVM_linear_training_score_list_LOGO = training_score
            SVM_linear_test_score_list_LOGO = test_score
            SVM_linear_matrix_LOGO = conf_matrix

            training_score,classifier, test_score, best_params, conf_matrix=SVM_rbf.leave_one_group_out()
            SVM_RBF_training_score_list_LOGO = training_score
            SVM_RBF_test_score_list_LOGO = test_score
            SVM_RBF_matrix_LOGO = conf_matrix
            with open(filename, "a") as f:
                f.write("Best parameters of the {} after gridSearch with LOGO validation are:\n{}".format(classifier,best_params))
                f.write(f"The training score is {training_score}\n")
                f.write(f"The confusion matrix is: {conf_matrix}")
                f.write(f"Accuracy of the 10 fold Validation on the test set is {test_score}\n\n\n")
            # SVM_rbf.visualization_PCA()
            
            training_score,classifier, test_score, best_params, conf_matrix = KNeighbors.leave_one_group_out()
            #  KNeighbors.visualization_PCA()
            kNN_training_score_list_LOGO = training_score
            kNN_test_score_list_LOGO = test_score
            kNN_matrix_LOGO = conf_matrix 
            with open(filename, "a") as f:
                f.write("Best parameters of the {} after gridSearch with LOGO validation are:\n{}".format(classifier,best_params))
                f.write(f" The training score is {training_score}\n")
                f.write(f"The confusion matrix is: {conf_matrix}")
                f.write(f"Accuracy of the 10 fold Validation on the test set is {test_score}\n\n\n")
            
            training_score,classifier, test_score, best_params, conf_matrix = random_forest.leave_one_group_out()
            random_forest.visualization_PCA()
            RF_training_score_list_LOGO =  training_score
            RF_test_score_list_LOGO = test_score
            RF_matrix_LOGO  = conf_matrix
            with open(filename, "a") as f:
                f.write("Best parameters of the {} after gridSearch with LOGO validation are:\n{}".format(classifier,best_params))
                f.write(f" The training score is {training_score}\n")
                f.write(f"The confusion matrix is: {conf_matrix}")
                f.write(f"Accuracy of the 10 fold Validation on the test set is {test_score}\n\n\n")

        results_dict = dict()

        # result_keys_tenfold = ["SVM_linear_training_score_list_tenfold","SVM_linear_test_score_list_tenfold","SVM_linear_matrix_tenfold", 
                    
        #             "SVM_RBF_training_score_list_tenfold","SVM_RBF_test_score_list_tenfold", "SVM_RBF_matrix_tenfold",
                    
        #             "kNN_training_score_list_tenfold","kNN_test_score_list_tenfold","kNN_matrix_tenfold",
                    
        #             "RF_training_score_list_tenfold","RF_test_score_list_tenfold","RF_matrix_tenfold",
        #             
        #             ]
        result_keys_LOGO = ["SVM_linear_training_score_list_LOGO", "SVM_linear_test_score_list_LOGO", "SVM_linear_matrix_LOGO",
                            "SVM_RBF_training_score_list_LOGO", "SVM_RBF_test_score_list_LOGO", "SVM_RBF_matrix_LOGO", 
                            "kNN_training_score_list_LOGO", "kNN_test_score_list_LOGO", "kNN_matrix_LOGO",
                            "RF_training_score_list_LOGO","RF_test_score_list_LOGO","RF_matrix_LOGO"]
            
        # for key in result_keys_tenfold:
        #     with open(filename, "a") as f:
        #         f.write(f"{key}:{locals().get(key)},\n and the mean is {np.mean(locals().get(key))}\n\n")
        #     results_dict[key] = locals().get(key)
        
        for key in result_keys_LOGO:
            with open(filename, "a") as f:
                f.write(f"{key}:{locals().get(key)},,\n and the mean is {np.mean(locals().get(key))}\n\n")
            results_dict[key] = locals().get(key)

        torch.save(results_dict,picklename)



