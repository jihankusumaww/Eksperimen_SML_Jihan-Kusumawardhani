# 🏥 Advanced Machine Learning System: Insurance Data Preprocessing and Baseline Training Pipeline

## Project Overview

This project delivers a robust, end-to-end solution for preprocessing and Machine Learning model training on healthcare insurance data. The primary objective is to establish a pipeline that is fundamentally **reproducible**, meticulously documented, and readily prepared for deployment or subsequent experimentation. Every stage is automated, covering raw data cleaning, systematic outlier handling, feature encoding, scaling, model training, artifact logging via MLflow, and seamless integration with GitHub Actions for Continuous Integration (CI).

## 📚 Context and Motivation

In the domain of health insurance, accurate medical cost prediction is paramount for informed decision-making, premium calculation, and comprehensive risk analysis. The dataset utilized in this project is sourced from the widely-used [Kaggle - Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance). It contains personal medical cost data based on key features, including:

- **age**: Beneficiary's age
- **sex**: Gender
- **bmi**: Body Mass Index
- **children**: Number of children covered by health insurance
- **smoker**: Smoking status
- **region**: Residential area in the US (e.g., northeast, southwest)
- **charges**: Individual annual medical charges (Target Variable)

A well-engineered preprocessing workflow is critical to ensure that the Machine Learning model can learn optimally from the data, leading to reliable and trustworthy predictions.

## 🎯 Project Objectives

1.  To provide an automated, consistent, and highly reusable preprocessing pipeline.
2.  To systematically address data inconsistencies, including missing values and statistical outliers.
3.  To ensure the entire workflow is executable both locally and automatically via GitHub Actions (Continuous Integration).
4.  To log all critical artifacts (pipeline object, processed data, trained model, and metrics) into **MLflow** for robust experiment tracking and guaranteed reproducibility.
5.  To facilitate smooth collaboration and simplified deployment through clear documentation and standardized workflows.

## ✨ Core Features and Technical Implementation

- ⚡ **Automated Preprocessing:** Unified, end-to-end transformation for both numerical and categorical features, eliminating the need for manual, step-by-step transformations.
- 🧹 **Data Cleansing:** Systematic imputation/removal of missing values, ensuring the dataset is primed for effective modeling.
- 🚨 **Outlier Management:** Application of the **IQR (Interquartile Range)** method on numerical features to mitigate model bias induced by extreme data points.
- 📏 **MinMax Scaling:** Standardized scaling of numerical features to ensure data uniformity, thereby accelerating model convergence.
- 🏷️ **One-Hot Encoding:** Transformation of categorical variables, enabling their utilization by both linear and non-linear models.
- 🔀 **Automated Data Splitting:** Consistent 70:30 train-test split, ensuring an unbiased model evaluation framework.
- 💾 **Pipeline Persistence:** The full preprocessing pipeline is serialized (using `joblib`) for reuse on new, unseen data without requiring retraining.
- 📊 **Baseline Modeling:** Implementation of a **Linear Regression model** to establish a performance benchmark for medical cost prediction.
- 📝 **MLflow Artifact Logging:** Comprehensive logging of the entire transformation pipeline, resulting datasets, the final model, and performance metrics, guaranteeing **traceability** of all experiments.
- 🤖 **GitHub Actions CI/CD Workflow:** Automation of the entire pipeline execution in the cloud upon code changes, establishing a standard for Continuous Integration.

## 🗂️ Project Structure

The directory structure is organized for clear navigation and maintenance:

```
.
├── insurance_raw.csv                          # Dataset mentah
├── run_preprocessing.py                       # Script utama
├── requirements.txt                           # Dependencies Python
├── preprocessing/
│   ├── automate_jkw.py                       # Modul preprocessing
│   ├── preprocessor.joblib                   # Pipeline tersimpan
│   └── insurance_preprocessing/
│       ├── columns.csv                       # Header kolom
│       ├── insurance_train_preprocessed.csv  # Data training
│       └── insurance_test_preprocessed.csv   # Data testing
└── .github/
    └── workflows/
        └── preprocessing.yml                 # GitHub Actions workflow
        
```

## 🚀 Execution Guide (Step-by-Step)

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPO.git](https://github.com/YOUR_USERNAME/YOUR_REPO.git)
    cd YOUR_REPO
    ```

2.  **Acquire the Dataset:**
    - Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/mirichoi0218/insurance).
    - Download `insurance.csv`.
    - Rename the file to `insurance_raw.csv` and place it in the project root directory.

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run Pipeline Execution:**
    ```bash
    python run_preprocessing.py
    ```
    - *This single command executes all steps: data cleaning, outlier removal, encoding, scaling, data split, baseline training, and MLflow logging.*

5.  **Verify Artifacts:**
    - 📂 Processed data and the pipeline object will be generated in `preprocessing/insurance_preprocessing/`.
    - 📈 If the MLflow tracking server is running, all artifacts, metrics, and run parameters will be accessible via the MLflow UI.

6.  **(Optional) Automatic CI/CD Execution:**
    - Any push or pull request to the main branch will automatically trigger the GitHub Actions workflow.
    - Resulting artifacts are available for download from the GitHub Actions interface.

### Example Use Cases

-   **Data Science Experimentation:** Rapidly iterate on the preprocessing steps, re-run the pipeline, and compare performance metrics directly within the MLflow UI.
-   **Production Deployment:** The serialized pipeline (`preprocessor.joblib`) and the model are ready for immediate integration into a production serving application.
-   **Team Collaboration:** Guarantees that all team members execute the identical, verifiable data transformation and training process.

## 🤖 GitHub Actions CI Workflow

This pipeline features integrated CI via GitHub Actions. Upon relevant code changes (push/pull requests) to the main branch, the automated workflow will:

-   Execute the preprocessing and training script.
-   Save resulting artifacts (processed data, pipeline, model) as downloadable GitHub Artifacts.
-   Ensure pipeline integrity and functional stability.

The workflow can also be triggered manually via the GitHub Actions menu. Artifacts are stored temporarily for 30 days.

## 📦 Pipeline Output Artifacts

Upon successful execution, the following critical artifacts are generated for downstream use, analysis, and reporting:

-   `preprocessor.joblib`: The reusable, persisted preprocessing pipeline object.
-   `insurance_train_preprocessed.csv`: The clean, processed data ready for training.
-   `insurance_test_preprocessed.csv`: The clean, processed data reserved for model evaluation.
-   `columns.csv`: Reference file detailing the final processed feature set.
-   `model.pkl`/`MLmodel`: The trained baseline model (logged in MLflow, ready for deployment).
-   📊 All artifacts, metrics, and the complete history of the run are recorded in the **MLflow UI**.

These artifacts ensure the entire process is fully repeatable and verifiable at any time.

## 👩‍💻 Author

This project was developed by:

**Jihan Kusumawardhani**

## 📄 License

This project is licensed under the MIT License. You are free to use, modify, and distribute the code, provided appropriate attribution is maintained.