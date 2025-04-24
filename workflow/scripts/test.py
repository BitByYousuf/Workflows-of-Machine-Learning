import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    with open(snakemake.output[0], "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"f1 Score: {f1:.4f}\n")

def main():
    model = joblib.load(snakemake.input.model)
    X_test, y_test = joblib.load(snakemake.input.test_data)

    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()