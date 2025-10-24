# BinaryClassifierML

## Project Overview
This project implements a machine learning pipeline for binary classification, demonstrated using the UCI Adult Census dataset but designed to support any binary classification dataset. It includes exploratory data analysis (EDA), data preprocessing, model training, evaluation, and an interactive Streamlit web application with SHAP model explanations.

### Key Features
- **Dataset**: Supports any dataset with a binary target; includes a demo with the UCI Adult Census dataset
- **Preprocessing**: Handles missing values, encodes categorical features, supports feature engineering, normalization, and outlier removal
- **Models**: Logistic Regression, K-Nearest Neighbors (KNN), SVM, Decision Tree, Random Forest, and Gradient Boosting
- **Evaluation**: Accuracy, Precision, Recall, F1-Score, AUC-ROC; visualizations include confusion matrices, ROC curves, and feature importance
- **Web App**: Streamlit-based interface for model comparison, predictions, and SHAP explanations
- **Author**: Siwar Haddad

## Repository Structure
- `Projet_Challenge_Siwar_Haddad.ipynb`: Main Jupyter notebook with the complete ML pipeline
- `app.py`: Streamlit web application script
- `adult_preprocessed.csv`: Preprocessed UCI Adult dataset (generated during notebook execution)
- `model_results.csv`: Model performance metrics (generated during notebook execution)
- `*_model.pkl`: Trained model files (e.g., `logistic_regression_model.pkl`, `random_forest_model.pkl`)
- `requirements.txt`: Python dependencies for the project
- `README.md`: This file

## Getting Started

### Prerequisites
- Python 3.8+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Node.js and npm for localtunnel (optional, for web app external access)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/BinaryClassifierML.git
   cd BinaryClassifierML
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   npm install localtunnel
   ```
3. For the demo, ensure the UCI Adult dataset is accessible (loaded via URL in the notebook). For custom datasets, upload a CSV file via the web app.

### Running the Notebook
1. Open `Projet_Challenge_Siwar_Haddad.ipynb` in Jupyter Notebook or Google Colab.
2. Execute cells sequentially to:
   - Load and preprocess data
   - Perform EDA
   - Train models
   - Save preprocessed data and model results
3. Outputs generated:
   - `adult_preprocessed.csv`: Preprocessed dataset (for UCI Adult demo)
   - `model_results.csv`: Model performance metrics
   - Model files: `logistic_regression_model.pkl`, `random_forest_model.pkl`, etc.
   - Visualizations: Printed in the notebook (confusion matrices, ROC curves, feature importance)

### Running the Streamlit Web App
1. Ensure the notebook has been run to generate `adult_preprocessed.csv`, `model_results.csv`, and model `.pkl` files (for the demo dataset).
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. For external access (e.g., on Colab):
   ```bash
   wget -q -O - ipv4.icanhazip.com
   npx localtunnel --port 8501
   ```
   - Copy the IP address and use the provided localtunnel URL to access the app.

#### Web App Functionalities
The Streamlit web application (`app.py`) provides an interactive interface for exploring datasets, models, and predictions. Key functionalities include:

- **Dataset Configuration** (Sidebar):
  - **Default Dataset**: Load the preprocessed UCI Adult dataset (`adult_preprocessed.csv`) with a single click.
  - **Custom Dataset Upload**: Upload a custom CSV file, select a binary target column, and trigger model training.
  - **Dataset Info**: View dataset shape, missing values, data types, and a preview of the first 10 rows (for custom datasets).

- **Model Performance Comparison**:
  - **Metrics Table**: Displays Accuracy, Precision, Recall, F1-Score, and AUC-ROC for all models, with color-coded gradients for easy comparison.
  - **Best Model Highlight**: Shows the model with the highest AUC-ROC and average accuracy across models.

- **ROC Curves Comparison**:
  - Interactive Plotly chart comparing ROC curves for all models, with AUC scores in the legend.
  - Includes a dashed line for a random classifier baseline.

- **Prediction Interface** (Sidebar):
  - **Model Selection**: Choose from trained models (e.g., Logistic Regression, Random Forest).
  - **Dynamic Feature Input**: Adjust up to 10 features using sliders (for numerical features) or default values (for one-hot encoded features).
  - **Prediction Output**: Displays predicted class (e.g., Class 0 or Class 1), confidence score, and model used.
  - **Probability Distribution**: Bar chart showing class probabilities (Class 0 and Class 1).
  - **SHAP Explanations**: Waterfall plot showing feature contributions to the prediction, using `shap.TreeExplainer` for tree-based models or `shap.KernelExplainer` for others.

- **Detailed Model Analysis** (Tabs):
  - **Confusion Matrix**: Select a model to view a heatmap of true vs. predicted labels, with metrics for True Negatives, False Positives, False Negatives, and True Positives.
  - **Classification Report**: Detailed metrics (precision, recall, F1-score) per class for a selected model, with a color-coded table.
  - **Precision-Recall Curve**: Interactive Plotly curve for a selected model, showing the trade-off between precision and recall.
  - **Feature Importance**: Bar chart of the top 20 features for Random Forest or Gradient Boosting, based on model feature importances.

- **User Guidance**:
  - Footer with step-by-step instructions on using the app, including dataset selection, model training, predictions, and analysis.
  - Error handling for missing files, invalid datasets, or non-binary target columns.

## Outputs
The following files are generated during notebook execution (for the UCI Adult demo):
- **Data**:
  - `adult_preprocessed.csv`: Cleaned and encoded dataset with features like `capital_net` and one-hot encoded categorical variables.
- **Models**:
  - `logistic_regression_model.pkl`
  - `random_forest_model.pkl`
  - `knn_model.pkl`
  - `svm_model.pkl`
  - `decision_tree_model.pkl`
- **Results**:
  - `model_results.csv`: Table with model performance metrics (Accuracy, Precision, Recall, F1-Score, AUC-ROC)
- **Visualizations** (displayed in notebook):
  - Boxplots for numerical features (outlier detection)
  - Correlation heatmaps and pairplots (EDA)
  - Confusion matrices for each model
  - ROC curves comparing all models
  - Feature importance plots for Decision Tree and Random Forest
  - SHAP waterfall plots (in web app for predictions)

## Usage
1. **Exploratory Data Analysis**:
   - Run sections 3.1–3.4 to understand dataset structure, distributions, and correlations.
2. **Preprocessing**:
   - Sections 4.1–4.7 handle missing values, encoding, feature engineering, normalization, and outlier removal.
3. **Model Training and Evaluation**:
   - Section 5 trains multiple classifiers, evaluates performance, and generates visualizations.
4. **Web App**:
   - Section 6 launches an interactive Streamlit app for predictions and model analysis.
   - Use the sidebar to select the dataset, train models (if using custom data), and input feature values for predictions.

## Key Insights
- **Data**: The demo dataset (UCI Adult) has 32,561 rows and 15 features, with some missing values in `workclass`, `occupation`, and `native_country`.
- **Preprocessing**: Features like `capital_gain` and `capital_loss` are combined into `capital_net`, and outliers are capped or removed (for demo dataset).
- **Model Performance**: Random Forest and Gradient Boosting typically outperform others in AUC-ROC due to their ability to handle complex relationships.
- **SHAP Explanations**: The web app provides feature importance for individual predictions, highlighting key drivers for the selected dataset.

## Notes
- The notebook supports any binary classification dataset via the web app, with the UCI Adult dataset as a demo.
- Some cells (e.g., outlier removal, model training) may take time due to dataset size or computational complexity.
- The Streamlit app requires pre-trained models and preprocessed data for the demo dataset, or user-uploaded data with model training for custom datasets.
- SHAP explanations may be slower for non-tree-based models (e.g., Logistic Regression, KNN) due to KernelExplainer.

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/new-feature`).
3. Commit changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- UCI Machine Learning Repository for the Adult Census dataset
- Streamlit and SHAP communities for excellent tools
- Scikit-learn for robust ML algorithms