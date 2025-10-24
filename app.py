import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn import svm

import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(page_title="Adult Census Income Prediction with SHAP", layout="wide")

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

# Title and description
st.title("🎯 Adult Census Income Prediction with Model Interpretation")
st.markdown("""
This app predicts income (>50K or <=50K) using machine learning models and provides SHAP explanations.
Use the sidebar to configure the dataset, train models, and make predictions.
""")

# Sidebar for dataset selection and upload
st.sidebar.header("📊 Dataset Configuration")
use_default = st.sidebar.checkbox("Use Default Adult Dataset", value=True)

# Function to load default data
@st.cache_data
def load_default_data():
    try:
        df_default = pd.read_csv('adult_preprocessed.csv')
        results_default = pd.read_csv('model_results.csv')
        return df_default, results_default, True
    except FileNotFoundError as e:
        st.error(f"❌ Error loading default dataset: {e}")
        return None, None, False

# Function to load models
@st.cache_resource
def load_default_models():
    try:
        models_dict = {
            'Logistic Regression': pickle.load(open('logistic_regression_model.pkl', 'rb')),
            'K-Nearest Neighbors': pickle.load(open('knn_model.pkl', 'rb')),
            'SVM': pickle.load(open('svm_model.pkl', 'rb')),
            'Random Forest': pickle.load(open('random_forest_model.pkl', 'rb')),
            'Decision Tree': pickle.load(open('decision_tree_model.pkl', 'rb')),
        }
        return models_dict, True
    except FileNotFoundError as e:
        st.error(f"❌ Error loading models: {e}")
        return {}, False

# Function to process and train on uploaded dataset
@st.cache_data
def process_and_train_uploaded(df, target_column):
    try:
        # Check if target is binary
        if df[target_column].nunique() != 2:
            st.error("Target feature must be binary (2 unique values)")
            return None, None, None, None, None, {}

        # Label encode target
        le = LabelEncoder()
        y = le.fit_transform(df[target_column])
        target_classes = le.classes_

        X = df.drop(target_column, axis=1)

        # Identify numerical and categorical columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Preprocessing pipeline
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])

        # Fit and transform
        X_preprocessed = preprocessor.fit_transform(X)

        # Get feature names
        feature_names = numerical_cols.copy()
        if categorical_cols:
            cat_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)
            feature_names.extend(cat_features.tolist())

        X_preprocessed = pd.DataFrame(X_preprocessed, columns=feature_names)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_preprocessed, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train models
        models_dict = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7),
            'SVM': svm.SVC(kernel='linear', C=1, random_state=42, probability=True),
            'Decision Tree': tree.DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }

        results_list = []
        trained_models = {}

        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, (name, model) in enumerate(models_dict.items()):
            status_text.text(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            # Metrics
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_score = auc(fpr, tpr)

            results_list.append({
                'Model': name,
                'Accuracy': report['accuracy'],
                'Precision': report['weighted avg']['precision'],
                'Recall': report['weighted avg']['recall'],
                'F1-Score': report['weighted avg']['f1-score'],
                'AUC-ROC': auc_score
            })

            trained_models[name] = model
            progress_bar.progress((idx + 1) / len(models_dict))

        status_text.text("✅ Training complete!")
        progress_bar.empty()
        status_text.empty()

        results_df = pd.DataFrame(results_list)

        # Store train/test split for later use
        split_data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_preprocessed': X_preprocessed,
            'y': y
        }

        return X_preprocessed, y, results_df, trained_models, split_data, {
            'numerical_cols': numerical_cols,
            'categorical_cols': categorical_cols,
            'target_classes': target_classes
        }
    except Exception as e:
        st.error(f"❌ Error processing dataset: {str(e)}")
        return None, None, None, None, None, {}

# Load data based on selection
if use_default:
    df, results, success = load_default_data()
    if success:
        models, model_success = load_default_models()
        if model_success:
            st.sidebar.success("✅ Default dataset and models loaded")
            target_column = 'income'
            st.session_state.data_loaded = True
            st.session_state.models_trained = True

            # Create split data for default dataset
            X = df.drop(target_column, axis=1)
            y = df[target_column]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            split_data = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'X_preprocessed': X,
                'y': y
            }
            metadata = {'target_classes': [0, 1]}
        else:
            st.stop()
    else:
        st.stop()
else:
    uploaded_file = st.sidebar.file_uploader("📁 Upload CSV dataset", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

            # Display dataset info
            with st.expander("📋 View Dataset Information"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", df.shape[0])
                with col2:
                    st.metric("Columns", df.shape[1])
                with col3:
                    st.metric("Missing Values", df.isnull().sum().sum())

                st.subheader("Data Preview")
                st.dataframe(df.head(10))

                st.subheader("Data Types")
                st.dataframe(pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.astype(str),
                    'Non-Null': df.count().values,
                    'Unique': [df[col].nunique() for col in df.columns]
                }))

            # Choose target feature
            target_column = st.sidebar.selectbox(
                "🎯 Select Target Feature (binary)",
                df.columns,
                help="Choose the column you want to predict (must have exactly 2 unique values)"
            )

            if target_column:
                st.sidebar.info(f"Target: {target_column}\nUnique values: {df[target_column].nunique()}")

                if st.sidebar.button("🚀 Train Models", type="primary"):
                    with st.spinner("Training models..."):
                        X, y, results, models, split_data, metadata = process_and_train_uploaded(df, target_column)
                        if X is not None:
                            df = pd.concat([X, pd.Series(y, name=target_column)], axis=1)
                            st.session_state.data_loaded = True
                            st.session_state.models_trained = True
                            st.sidebar.success("✅ Models trained successfully!")
                        else:
                            st.stop()
                else:
                    st.info("👈 Please click 'Train Models' in the sidebar to begin")
                    st.stop()
        except Exception as e:
            st.error(f"❌ Error loading dataset: {str(e)}")
            st.stop()
    else:
        st.info("👈 Please upload a CSV file to begin")
        st.stop()

# Check if data is loaded
if not st.session_state.data_loaded:
    st.warning("⚠️ No dataset loaded. Please configure the dataset in the sidebar.")
    st.stop()

# Model Performance Section
st.header("📊 Model Performance Comparison")
if results is not None and not results.empty:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.dataframe(
            results.style.format({
                'Accuracy': '{:.2%}',
                'Precision': '{:.2%}',
                'Recall': '{:.2%}',
                'F1-Score': '{:.2%}',
                'AUC-ROC': '{:.4f}'
            }).background_gradient(cmap='RdYlGn', subset=['Accuracy', 'AUC-ROC']),
            use_container_width=True
        )

    with col2:
        best_model = results.loc[results['AUC-ROC'].idxmax(), 'Model']
        best_auc = results.loc[results['AUC-ROC'].idxmax(), 'AUC-ROC']
        st.metric("🏆 Best Model", best_model, f"AUC: {best_auc:.4f}")

        avg_accuracy = results['Accuracy'].mean()
        st.metric("📈 Average Accuracy", f"{avg_accuracy:.2%}")

# ROC Curves
st.header("📈 ROC Curves Comparison")
try:
    fig = go.Figure()
    for name in models:
        model = models[name]
        y_prob = model.predict_proba(split_data['X_test'])[:, 1]
        fpr, tpr, _ = roc_curve(split_data['y_test'], y_prob)
        auc_score = auc(fpr, tpr)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f'{name} (AUC = {auc_score:.3f})',
            mode='lines',
            line=dict(width=2)
        ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        line=dict(dash='dash', color='gray', width=1),
        name='Random Classifier',
        showlegend=True
    ))

    fig.update_layout(
        title="Receiver Operating Characteristic (ROC) Curves",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0.6, y=0.1),
        hovermode='closest',
        width=None,
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"Error generating ROC curves: {str(e)}")

# Prediction Section
st.header("🔮 Make Predictions")
st.sidebar.header("🎛️ Prediction Settings")

# Model selection
model_name = st.sidebar.selectbox("Select Model", list(models.keys()))

# Dynamic user input based on available features
st.sidebar.subheader("Input Features")

def get_user_input_dynamic(df, target_col):
    """Generate dynamic input based on available features"""
    input_data = {}
    feature_cols = [col for col in df.columns if col != target_col]

    # Limit to first 10 features for UI simplicity
    display_features = feature_cols[:min(10, len(feature_cols))]

    for col in display_features:
        if df[col].dtype in ['int64', 'float64']:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            mean_val = float(df[col].mean())
            input_data[col] = st.sidebar.slider(
                col.replace('_', ' ').title(),
                min_val, max_val, mean_val
            )
        else:
            # For one-hot encoded features, set to 0 by default
            input_data[col] = 0

    # Fill remaining features with 0
    for col in feature_cols:
        if col not in input_data:
            input_data[col] = 0

    return pd.DataFrame([input_data])[feature_cols]

user_input = get_user_input_dynamic(df, target_column)

# Prediction button
if st.sidebar.button("🎯 Predict", type="primary"):
    try:
        model = models[model_name]

        # Predict
        prediction = model.predict(user_input)[0]
        probability = model.predict_proba(user_input)[0]

        # Display results
        col1, col2, col3 = st.columns(3)

        with col1:
            result_label = "High Income (>50K)" if prediction == 1 else "Low Income (<=50K)"
            st.metric("Prediction", result_label)

        with col2:
            st.metric("Confidence", f"{probability[prediction]:.1%}")

        with col3:
            st.metric("Model", model_name)

        # Probability distribution
        fig_prob = go.Figure(data=[
            go.Bar(
                x=['Class 0', 'Class 1'],
                y=probability,
                marker_color=['#FF6B6B', '#4ECDC4'],
                text=[f'{p:.1%}' for p in probability],
                textposition='auto',
            )
        ])
        fig_prob.update_layout(
            title="Class Probability Distribution",
            yaxis_title="Probability",
            showlegend=False,
            height=300
        )
        st.plotly_chart(fig_prob, use_container_width=True)

        # SHAP Explanation
        st.subheader("🔍 SHAP Feature Importance Explanation")

        with st.spinner("Generating SHAP values..."):
            try:
                if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier)):
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(user_input)
                    base_value = explainer.expected_value
                    # Handle multi-output models
                    if isinstance(shap_values, list):
                        # For binary classification, select the positive class (index 1)
                        shap_values = shap_values[1]
                        base_value = base_value[1] if isinstance(base_value, (list, np.ndarray)) else base_value
                    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) > 2:
                        # For multi-output arrays, select the positive class (index 1)
                        shap_values = shap_values[:, :, 1]
                        base_value = base_value[1] if isinstance(base_value, (list, np.ndarray)) else base_value
                else:
                    # Use a smaller background dataset for KernelExplainer
                    background = shap.sample(split_data['X_train'], 50)
                    explainer = shap.KernelExplainer(model.predict_proba, background)
                    shap_values = explainer.shap_values(user_input)
                    base_value = explainer.expected_value
                    # Handle multi-output models
                    if isinstance(shap_values, list):
                        # For binary classification, select the positive class (index 1)
                        shap_values = shap_values[1]
                        base_value = base_value[1] if isinstance(base_value, (list, np.ndarray)) else base_value
                    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) > 2:
                        # For multi-output arrays, select the positive class (index 1)
                        shap_values = shap_values[:, :, 1]
                        base_value = base_value[1] if isinstance(base_value, (list, np.ndarray)) else base_value

                # Ensure shap_values is for a single instance
                if len(shap_values.shape) > 1:
                    if shap_values.shape[0] > 1:
                        shap_values = shap_values[0]  # Select first instance
                    else:
                        shap_values = shap_values[0]  # Flatten if only one instance

                # Ensure base_value is a scalar
                if isinstance(base_value, (list, np.ndarray)):
                    if len(base_value) > 1:
                        base_value = float(base_value[0])  # Select first value and convert to scalar
                    else:
                        base_value = float(base_value[0])  # Flatten to scalar
                else:
                    base_value = float(base_value)  # Ensure scalar

                # Waterfall plot
                st.write("**Feature Impact on Prediction:**")
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.plots.waterfall(
                    shap.Explanation(
                        values=shap_values,
                        base_values=base_value,
                        data=user_input.iloc[0],
                        feature_names=user_input.columns.tolist()
                    ),
                    show=False
                )
                st.pyplot(fig)
                plt.close()

            except Exception as e:
                st.warning(f"Could not generate SHAP explanation: {str(e)}")

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

# Additional Analysis
st.header("📊 Detailed Model Analysis")

analysis_tabs = st.tabs(["Confusion Matrix", "Classification Report", "Precision-Recall", "Feature Importance"])

with analysis_tabs[0]:
    selected_model_cm = st.selectbox("Select model for confusion matrix", list(models.keys()), key='cm_model')
    if st.button("Generate Confusion Matrix"):
        try:
            model = models[selected_model_cm]
            y_pred = model.predict(split_data['X_test'])

            cm = confusion_matrix(split_data['y_test'], y_pred)

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                       xticklabels=['Class 0', 'Class 1'],
                       yticklabels=['Class 0', 'Class 1'])
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            ax.set_title(f'Confusion Matrix - {selected_model_cm}')
            st.pyplot(fig)
            plt.close()

            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            tn, fp, fn, tp = cm.ravel()
            with col1:
                st.metric("True Negatives", tn)
            with col2:
                st.metric("False Positives", fp)
            with col3:
                st.metric("False Negatives", fn)
            with col4:
                st.metric("True Positives", tp)
        except Exception as e:
            st.error(f"Error: {str(e)}")

with analysis_tabs[1]:
    selected_model_cr = st.selectbox("Select model for classification report", list(models.keys()), key='cr_model')
    if st.button("Generate Classification Report"):
        try:
            model = models[selected_model_cr]
            y_pred = model.predict(split_data['X_test'])

            report = classification_report(split_data['y_test'], y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()

            st.dataframe(
                report_df.style.format("{:.3f}").background_gradient(cmap='RdYlGn'),
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error: {str(e)}")

with analysis_tabs[2]:
    selected_model_pr = st.selectbox("Select model for precision-recall curve", list(models.keys()), key='pr_model')
    if st.button("Generate Precision-Recall Curve"):
        try:
            model = models[selected_model_pr]
            y_prob = model.predict_proba(split_data['X_test'])[:, 1]

            precision, recall, _ = precision_recall_curve(split_data['y_test'], y_prob)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=recall, y=precision,
                mode='lines',
                name='Precision-Recall Curve',
                fill='tozeroy',
                line=dict(color='#4ECDC4', width=2)
            ))
            fig.update_layout(
                title=f"Precision-Recall Curve - {selected_model_pr}",
                xaxis_title="Recall",
                yaxis_title="Precision",
                hovermode='closest',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {str(e)}")

with analysis_tabs[3]:
    selected_model_fi = st.selectbox("Select model for feature importance",
                                     [m for m in models.keys() if m in ['Random Forest', 'Gradient Boosting']],
                                     key='fi_model')
    if selected_model_fi and st.button("Show Feature Importance"):
        try:
            model = models[selected_model_fi]
            if hasattr(model, 'feature_importances_'):
                feature_imp = pd.DataFrame({
                    'Feature': split_data['X_test'].columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False).head(20)

                fig = px.bar(feature_imp, x='Importance', y='Feature', orientation='h',
                            title=f'Top 20 Feature Importances - {selected_model_fi}')
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Selected model does not support feature importances")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Footer with instructions
st.markdown("---")
st.markdown("""
### 📖 How to Use This App

1. **Dataset Selection**: Choose default dataset or upload your own CSV file
2. **Model Training**: If using custom data, click 'Train Models' to train classifiers
3. **View Performance**: Compare model metrics, ROC curves, and performance statistics
4. **Make Predictions**: Adjust input features in sidebar and click 'Predict'
5. **Analyze Results**: Explore confusion matrices, classification reports, and feature importance

**Supported Models**: Logistic Regression, Random Forest, Gradient Boosting, K-Nearest Neighbors

**SHAP Explanations**: Understand which features contribute most to each prediction
""")
