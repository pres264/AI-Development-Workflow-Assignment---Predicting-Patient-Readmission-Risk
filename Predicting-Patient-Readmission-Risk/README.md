# AI-Development-Workflow-Assignment---Predicting-Patient-Readmission-Risk

**Course:** AI for Software Engineering   
**Team Members:** Presley Oluoch, Daisy Achieng, Elias Derick, Moses Kerama, Berlyn Mumbua.  

## 📌 Overview
This repository contains our solution for the AI Development Workflow assignment, where we develop a system to predict 30-day patient readmission risk for a hospital. The project follows the complete AI development lifecycle from problem definition to deployment considerations.

## 🗂️ Project Structure

- **src/**: Main source code files.
  - `data_preprocessing.py`: Data cleaning, encoding, and scaling functions.
  - `model_training.py`: Model training, hyperparameter tuning, and saving.
  - `evaluation.py`: Model evaluation and reporting.
  - `utils.py`: Utility functions for data/model loading and printing.
- **notebooks/**: Jupyter notebooks for EDA and pipeline prototyping.
  - `preprocessing_pipeline.ipynb`: Preprocessing, feature engineering, and visualization.
- **reports/**: Project report.
  - `AI_Development_Workflow_Report.pdf`: Full write-up, including workflow, results, and ethical analysis.
- **diagrams/**: Visual workflow representations.
  - `workflow_diagram.png`: AI development workflow diagram.
  - `workflow_diagram.pptx`: Editable diagram.
- **data/**: Sample and processed data.
  - `sample_data.csv`: Simulated patient records.
  - `cleaned_data.csv`: Cleaned and preprocessed data (generated).
  - `patient_readmission_model.pkl`: Trained model (generated).
  - `evaluation_report.txt`: Model evaluation report (generated).
- **references/**: Research references.
  - `research_references.md`: All sources used.
- **requirements.txt**: Python dependencies.

## 🚀 Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Preprocess the data
```bash
python src/data_preprocessing.py
```

### 3. Train the model
```bash
python src/model_training.py
```

### 4. Evaluate the model
```bash
python src/evaluation.py
```

### 5. (Optional) Explore the notebook
Open `notebooks/preprocessing_pipeline.ipynb` in Jupyter for EDA and pipeline steps.

## 🛠️ Troubleshooting
- Ensure your working directory is the project root.
- If you encounter file not found errors, check the relative paths in the scripts.
- All generated files (cleaned data, model, evaluation report) will appear in the `data/` directory.

## 🧭 Workflow Diagram
See `diagrams/workflow_diagram.png` for a visual overview of the AI development process.

## 📄 Supporting Materials
- Compiled PDF report (5-10 pages)
- Sample data examples
- Research references

For any questions or issues, please contact the team members listed above.