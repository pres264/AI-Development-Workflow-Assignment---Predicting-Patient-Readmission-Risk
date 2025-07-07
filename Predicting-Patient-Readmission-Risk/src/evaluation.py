import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import os

def load_model(model_path):
    """
    Load a trained model from disk.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)

def load_test_data(data_path, target_col):
    """
    Load cleaned data and split into features and target.
    """
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target_col, 'patient_id']) if 'patient_id' in df.columns else df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def evaluate(model, X_test, y_test, output_path=None):
    """
    Evaluate the model and print/save metrics.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    if output_path:
        with open(output_path, 'w') as f:
            f.write("Confusion Matrix:\n" + str(cm) + "\n\n")
            f.write("Classification Report:\n" + report + "\n")
            f.write(f"Precision: {precision:.2f}\n")
            f.write(f"Recall: {recall:.2f}\n")
            f.write(f"F1 Score: {f1:.2f}\n")
        print(f"Evaluation results saved to {output_path}")

if __name__ == "__main__":
    model = load_model('../data/patient_readmission_model.pkl')
    X, y = load_test_data('../data/cleaned_data.csv', 'readmission')
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    evaluate(model, X_test, y_test, output_path='../data/evaluation_report.txt')