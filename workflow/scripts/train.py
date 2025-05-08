import pandas as pd
from scipy.io import arff
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# Configure logging using Snakemake's log file
logging.basicConfig(
    filename=snakemake.log[0],  # Access first log file from Snakemake
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def split_data(data):
    X = data.drop('class', axis=1)
    y = data['class']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    logging.info("Training Gradient Boosting Classifier...")
    model = GradientBoostingClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    logging.info("Training complete.")
    return model

def main():
    data_path = snakemake.input[0]
    output_path = snakemake.output[0]
    output_path_data = snakemake.output[1]

    logging.info(f"Loading data from {data_path}")
    data, meta = arff.loadarff(data_path)
    data = pd.DataFrame(data)

    logging.info(f"Loaded dataset with shape: {data.shape}")

    X_train, X_test, y_train, y_test = split_data(data)
    logging.info(f"Split data into {len(X_train)} training and {len(X_test)} testing samples.")

    model = train_model(X_train, y_train)

    joblib.dump(model, output_path)
    logging.info(f"Saved trained model to {output_path}")

    joblib.dump((X_test, y_test), output_path_data)
    logging.info(f"Saved test data to {output_path_data}")

if __name__ == "__main__":
    main()
