# AI-Development-Workflow-Assignment---Predicting-Patient-Readmission-Risk

**Course:** AI for Software Engineering  
**Team Members:** Presley Oluoch, Daisy Achieng, Elias Derick, Moses Kerama, Berlyn Mumbua.

## 📌 Project Overview
This repository contains a complete AI development workflow for predicting 30-day patient readmission risk in a hospital setting. The project demonstrates best practices in data preprocessing, model development, evaluation, and deployment considerations, following a real-world healthcare case study.

## 🗂️ Project Structure
- **src/**: Main source code for data preprocessing, model training, evaluation, and utilities.
- **notebooks/**: Jupyter notebook for exploratory data analysis, feature engineering, and pipeline prototyping.
- **data/**: Sample and processed data, as well as model and evaluation outputs.
- **diagrams/**: Workflow diagram (PNG and PPTX) visualizing the AI development process.
- **reports/**: Final PDF report summarizing the workflow, results, and ethical considerations.
- **references/**: Research references and sources used.
- **requirements.txt**: Python dependencies for reproducibility.

## 🚀 Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full workflow
```bash
python main.py
```
This will preprocess the data, train the model, and evaluate it. Outputs are saved in the `data/` directory:
- `cleaned_data.csv`: Cleaned and preprocessed data
- `patient_readmission_model.pkl`: Trained model
- `evaluation_report.txt`: Model evaluation metrics

### 3. Explore the notebook
Open `notebooks/preprocessing_pipeline.ipynb` in Jupyter for step-by-step EDA, feature engineering, and pipeline demonstration.

## 🧭 Workflow Diagram
See `diagrams/workflow_diagram.png` for a visual overview of the AI development process.

## 📄 Deliverables
- **PDF Report**: Full write-up in `reports/AI_Development_Workflow_Report.pdf`
- **Codebase**: All scripts and notebook in this repository
- **Workflow Diagram**: PNG and PPTX in `diagrams/`
- **References**: See `references/research_references.md`

## 🛠️ Troubleshooting
- Ensure your working directory is the project root.
- If you encounter file not found errors, check the relative paths in the scripts.
- All generated files (cleaned data, model, evaluation report) will appear in the `data/` directory.

For any questions or issues, please contact the team members listed above.
