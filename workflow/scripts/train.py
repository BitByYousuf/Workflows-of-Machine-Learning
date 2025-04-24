import pandas as pd
from scipy.io import arff
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

def split_data(data):
    X = data.drop('class', axis=1)
    y = data['class']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = GradientBoostingClassifier(n_estimators=150, random_state=42)
    
    print("Training Gradient Boosting Classifier...")
    model.fit(X_train, y_train)
    return model

def main():

    data_path = snakemake.input[0]
    output_path = snakemake.output[0]
    output_path_data = snakemake.output[1]

    data, meta = arff.loadarff(data_path)

    data = pd.DataFrame(data)

    X_train, X_test, y_train, y_test = split_data(data)

    model = train_model(X_train, y_train)

    # save the model and test set
    joblib.dump(model, output_path)
    joblib.dump((X_test, y_test), output_path_data)
    print("Model and test data saved.")

if __name__ == "__main__":
    main()