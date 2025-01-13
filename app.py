# APP
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
@st.cache
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Preprocessing function
def preprocess_data(df, target_column):
    df.fillna(df.median(), inplace=True)
    for column in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

# Train model function
def train_model(model, X_train, y_train, params=None):
    if params:
        model.set_params(**params)
    model.fit(X_train, y_train)
    return model

# Main app
def main():
    st.title("Divorce Prediction Model")
    st.sidebar.title("Settings")

    # File upload
    file_path = st.sidebar.file_uploader("Upload CSV", type="csv")
    if file_path:
        df = load_data(file_path)
        st.write("Dataset Preview")
        st.write(df.head())

        target_column = st.sidebar.selectbox("Select Target Column", df.columns)
        if st.sidebar.button("Preprocess Data"):
            X, y = preprocess_data(df, target_column)

            # SMOTE
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

            st.write("Data preprocessed and split.")

            # Model selection
            model_choice = st.sidebar.selectbox("Select Model", ["Logistic Regression", "Naive Bayes", "Random Forest"])
            if model_choice == "Logistic Regression":
                params = {"C": 1, "solver": "liblinear"}
                model = LogisticRegression()
            elif model_choice == "Naive Bayes":
                params = {"alpha": 1.0}
                model = MultinomialNB()
            elif model_choice == "Random Forest":
                params = {"n_estimators": 100, "max_depth": None}
                model = RandomForestClassifier(random_state=42)

            if st.sidebar.button("Train Model"):
                model = train_model(model, X_train, y_train, params)
                y_pred = model.predict(X_test)
                st.write("Model trained!")

                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                st.write(f"Accuracy: {accuracy:.2f}")
                st.write("Classification Report")
                st.json(report)

                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title(f"Confusion Matrix - {model_choice}")
                st.pyplot(fig)

    st.sidebar.write("Developed by [Your Name]")

if __name__ == "__main__":
    main()
