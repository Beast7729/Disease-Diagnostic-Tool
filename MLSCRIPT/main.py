from json import encoder
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import Counter
import seaborn as sns

class DiseasePredictor:
    def __init__(self, data_path, test_path):
        self.data_path = data_path
        self.test_path = test_path
        self.models = {
            "SVC": SVC(),
            "GaussianNB": GaussianNB(),
            "RandomForest": RandomForestClassifier(random_state=18),
            "DecisionTree": DecisionTreeClassifier(random_state=18)
        }

        self.encoder = LabelEncoder()
        self.data_dict = {}
        self.symptom_index = {}
        self.final_models = {}

    


# Define a function to get disease details
    def get_disease_info(self,disease_name):
        ddf=pd.read_csv('diseases.csv')
        # Check if the disease exists in the dataset
        disease_data = ddf[ddf['Disease'].str.lower() == disease_name.lower()]
        
        if not disease_data.empty:
            # Extract the relevant information
            symptoms = disease_data['Symptoms'].values[0]
            treatment = disease_data['Treatment'].values[0]
            fatality = disease_data['Fatality'].values[0]
            time_to_cure = disease_data['Time to Cure (Days/Weeks)'].values[0]
            
            # Return the disease details
            return {
                "Symptoms": symptoms,
                "Treatment": treatment,
                "Fatality": fatality,
                "Time to Cure": time_to_cure
            }
        else:
            return f"Disease '{disease_name}' not found in the database."




    def load_data(self):
        """Load the training data and prepare it."""
        self.data = pd.read_csv(self.data_path).dropna(axis=1)
        print(self.data['prognosis'].unique())
        


        self.data['prognosis'] = self.encoder.fit_transform(self.data['prognosis'])
        X = self.data.iloc[:, :-1]
        y = self.data.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X = X
        self.y = y
        
        print(f"Train: {X_train.shape}, {y_train.shape}")
        print(f"Test: {X_test.shape}, {y_test.shape}")
    

    def pltDatabase(self):
        DATA_PATH = "Training.csv"
        data = pd.read_csv(DATA_PATH).dropna(axis = 1)
        disease_counts = data["prognosis"].value_counts()
        temp_df = pd.DataFrame({
        "Disease": disease_counts.index.astype(str),
        "Counts": disease_counts.values
        })
        print(temp_df
              )
      
        plt.figure(figsize=(18,8))
        sns.barplot(x="Disease",y="Counts",data=temp_df)
        plt.xticks(rotation=90)
        plt.show()
    def cross_validation(self, cv=10):
        """Perform cross-validation for all models."""
        def cv_scoring(estimator, X, y):
            return accuracy_score(y, estimator.predict(X))
        
        for model_name, model in self.models.items():
            scores = cross_val_score(model, self.X, self.y, cv=cv, n_jobs=-1)
            print(f"{'=='*15}")
            print(model_name)
            print(f"Scores: {scores}")
            print(f"Mean Score: {np.mean(scores)}")
    

    def confusionmatrix(self):
        svm_model = SVC()
        svm_model.fit(self.X_train, self.y_train)
        preds = svm_model.predict(self.X_test)

        print(f"Accuracy on train data by SVM Classifier\
        : {accuracy_score(self.y_train, svm_model.predict(self.X_train))*100}")

        print(f"Accuracy on test data by SVM Classifier\
        : {accuracy_score(self.y_test, preds)*100}")
        cf_matrix = confusion_matrix(self.y_test, preds)
        plt.figure(figsize=(12,8))
        sns.heatmap(cf_matrix, annot=True)
        plt.title("Confusion Matrix for SVM Classifier on Test Data")
        plt.show()

        # Training and testing Naive Bayes Classifier
        nb_model = GaussianNB()
        nb_model.fit(self.X_train, self.y_train)
        preds = nb_model.predict(self.X_test)
        print(f"Accuracy on train data by Naive Bayes Classifier\
        : {accuracy_score(self.y_train, nb_model.predict(self.X_train))*100}")

        print(f"Accuracy on test data by Naive Bayes Classifier\
        : {accuracy_score(self.y_test, preds)*100}")
        cf_matrix = confusion_matrix(self.y_test, preds)
        plt.figure(figsize=(12,8))
        sns.heatmap(cf_matrix, annot=True)
        plt.title("Confusion Matrix for Naive Bayes Classifier on Test Data")
        plt.show()

        # Training and testing Random Forest Classifier
        rf_model = RandomForestClassifier(random_state=18)
        rf_model.fit(self.X_train, self.y_train)
        preds = rf_model.predict(self.X_test)
        print(f"Accuracy on train data by Random Forest Classifier\
        : {accuracy_score(self.y_train, rf_model.predict(self.X_train))*100}")

        print(f"Accuracy on test data by Random Forest Classifier\
        : {accuracy_score(self.y_test, preds)*100}")

        cf_matrix = confusion_matrix(self.y_test, preds)
        plt.figure(figsize=(12,8)),
        sns.heatmap(cf_matrix, annot=True)
        plt.title("Confusion Matrix for Random Forest Classifier on Test Data")
        plt.show()

    def train_models(self):
        """Train all models on the training data."""
        for model_name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            self.final_models[model_name] = model
            
    def evaluate_models(self):
        """Evaluate all models on the test data."""
        for model_name, model in self.final_models.items():
            preds = model.predict(self.X_test)
            print(f"Accuracy on train data by {model_name}: {accuracy_score(self.y_train, model.predict(self.X_train))*100}")
            print(f"Accuracy on test data by {model_name}: {accuracy_score(self.y_test, preds)*100}")
    
    def majority_voting(self):
        """Perform majority voting among the models to make final predictions."""
        test_data = pd.read_csv(self.test_path).dropna(axis=1)
        test_X = test_data.iloc[:, :-1]
        test_Y = self.encoder.transform(test_data.iloc[:, -1])
        
        # Get predictions from each model
        predictions = {model_name: model.predict(test_X) for model_name, model in self.final_models.items()}
        
        # Perform majority voting
        final_preds = [self.most_common([predictions['SVC'][i], predictions['GaussianNB'][i], predictions['RandomForest'][i], predictions['DecisionTree'][i]])
                       for i in range(len(test_X))]
        
        print(f"Accuracy on Test dataset by the combined model: {accuracy_score(test_Y, final_preds)*100}")
        
    def predict_disease(self, symptoms):
        """Predict disease based on input symptoms."""
        symptoms = symptoms.split(",")
        
        # Create input data for the models
        input_data = [0] * len(self.data_dict["symptom_index"])
        for symptom in symptoms:
            index = self.data_dict["symptom_index"][symptom]
            input_data[index] = 1
        
        input_data = np.array(input_data).reshape(1, -1)
        
        # Get individual predictions
        rf_prediction = self.data_dict["predictions_classes"][self.final_models["RandomForest"].predict(input_data)[0]]
        nb_prediction = self.data_dict["predictions_classes"][self.final_models["GaussianNB"].predict(input_data)[0]]
        svm_prediction = self.data_dict["predictions_classes"][self.final_models["SVC"].predict(input_data)[0]]
        dt_prediction = self.data_dict["predictions_classes"][self.final_models["DecisionTree"].predict(input_data)[0]]
        
        # Perform majority voting
        final_prediction = self.most_common([rf_prediction, nb_prediction, svm_prediction, dt_prediction])
        
        predictions = {
            "rf_model_prediction": rf_prediction,
            "naive_bayes_prediction": nb_prediction,
            "svm_model_prediction": svm_prediction,
            "dt_model_prediction": dt_prediction,
            "final_prediction": final_prediction
        }
        
        return predictions
    
    def create_symptom_index(self):
        """Create the symptom index for mapping symptoms to their respective columns."""
        symptoms = self.X.columns.values
        for index, value in enumerate(symptoms):
            symptom = " ".join([i.capitalize() for i in value.split("_")])
            self.symptom_index[symptom] = index
        self.data_dict = {
            "symptom_index": self.symptom_index,
            "predictions_classes": self.encoder.classes_
        }
    
    @staticmethod
    def most_common(lst):
        """Return the most common element from a list."""
        return Counter(lst).most_common(1)[0][0]

# Usage Example:
# disease_predictor = DiseasePredictor("Training.csv", "Testing.csv")
# disease_predictor.load_data()
# disease_predictor.cross_validation()
# disease_predictor.train_models()
# disease_predictor.evaluate_models()
# disease_predictor.majority_voting()
# disease_predictor.create_symptom_index()

# Testing disease prediction
# print(disease_predictor.predict_disease("Yellowish Skin,Yellowing Of Eyes,Belly Pain,Mild Fever,Chills"))
