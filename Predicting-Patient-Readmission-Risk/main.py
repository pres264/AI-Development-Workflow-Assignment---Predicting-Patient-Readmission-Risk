import os
from src.data_preprocessing import preprocess_and_save
from src.model_training import load_cleaned_data, split_data, train_and_tune, save_model
from src.evaluation import evaluate, load_model, load_test_data

def main():
    print("\nAI Development Workflow: Patient Readmission Risk Prediction\n")
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    raw_data = os.path.join(data_dir, 'sample_data.csv')
    cleaned_data = os.path.join(data_dir, 'cleaned_data.csv')
    model_path = os.path.join(data_dir, 'patient_readmission_model.pkl')
    eval_report = os.path.join(data_dir, 'evaluation_report.txt')

    # Step 1: Preprocess data
    print("Step 1: Preprocessing data...")
    preprocess_and_save(raw_data, cleaned_data, drop_cols=['patient_id'])

    # Step 2: Train model
    print("\nStep 2: Training model...")
    X, y = load_cleaned_data(cleaned_data, 'readmission')
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_and_tune(X_train, y_train)
    save_model(model, model_path)

    # Step 3: Evaluate model
    print("\nStep 3: Evaluating model...")
    evaluate(model, X_test, y_test, output_path=eval_report)
    print("\nWorkflow complete! See data/ for outputs.")

if __name__ == "__main__":
    main() 