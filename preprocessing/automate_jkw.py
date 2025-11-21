

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from joblib import dump
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow
import mlflow.sklearn

# Custom transformer untuk IQR outlier removal
class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        self.lower_bounds = {}
        self.upper_bounds = {}
        X_df = pd.DataFrame(X, columns=self.columns) if not isinstance(X, pd.DataFrame) else X
        for col in self.columns:
            Q1 = X_df[col].quantile(0.25)
            Q3 = X_df[col].quantile(0.75)
            IQR = Q3 - Q1
            self.lower_bounds[col] = Q1 - 1.5 * IQR
            self.upper_bounds[col] = Q3 + 1.5 * IQR
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=self.columns) if not isinstance(X, pd.DataFrame) else X.copy()
        mask = None
        for col in self.columns:
            col_mask = (X_df[col] >= self.lower_bounds[col]) & (X_df[col] <= self.upper_bounds[col])
            mask = col_mask if mask is None else (mask & col_mask)
        return X_df[mask].reset_index(drop=True)


def preprocess_data(data, target_column, save_path, file_path):
    mlflow.set_experiment("preprocessing-pipeline")
    with mlflow.start_run():
        numeric_features = data.select_dtypes(include=['float64','int64']).columns.tolist()
        categorical_features = data.select_dtypes(include=['object']).columns.tolist()
        column_names = data.columns.drop(target_column)

        # Membuat DataFrame kosong dengan nama kolom
        df_header = pd.DataFrame(columns=column_names)
        # Menyimpan nama kolom sebagai header tanpa data
        df_header.to_csv(file_path, index=False)
        print(f"Nama kolom berhasil disimpan ke: {file_path}")

        if target_column in numeric_features:
            numeric_features.remove(target_column)
        if target_column in categorical_features:
            categorical_features.remove(target_column)

        # Pipeline
        numeric_transformer = Pipeline(steps=[
            ('scaler', MinMaxScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('encoder', OneHotEncoder())
        ])
        preprocessor = ColumnTransformer(
            transformers = [
                ('num',numeric_transformer, numeric_features),
                ('cat',categorical_transformer, categorical_features)
            ]
        )

        # Memisahkan target
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Remove outlier dari data (hanya pada training set)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        outlier_remover = OutlierRemover(columns=numeric_features)
        Xy_train = X_train.copy()
        Xy_train[target_column] = y_train
        Xy_train_clean = outlier_remover.fit_transform(Xy_train)
        y_train_clean = Xy_train_clean[target_column]
        X_train_clean = Xy_train_clean.drop(columns=[target_column])

        # Fitting dan transformasi data pada training set
        X_train_final = preprocessor.fit_transform(X_train_clean)
        # Transformasi data pada testing set
        X_test_final = preprocessor.transform(X_test)

        # Simpan pipeline
        dump(preprocessor, save_path)

        # Simpan hasil preprocessing ke CSV
        train_csv_path = "preprocessing/insurance_preprocessing/insurance_train_preprocessed.csv"
        test_csv_path = "preprocessing/insurance_preprocessing/insurance_test_preprocessed.csv"
        pd.DataFrame(X_train_clean).assign(target=y_train_clean.reset_index(drop=True)).to_csv(train_csv_path, index=False)
        pd.DataFrame(X_test).assign(target=y_test.reset_index(drop=True)).to_csv(test_csv_path, index=False)

        # Pastikan file sudah dibuat sebelum log_artifact
        import os
        for f in [save_path, file_path, train_csv_path, test_csv_path]:
            if not os.path.exists(f):
                raise FileNotFoundError(f"File {f} tidak ditemukan sebelum log_artifact.")

        # Log artefak ke MLflow setelah file tersimpan
        mlflow.log_artifact(save_path)
        mlflow.log_artifact(file_path)
        mlflow.log_artifact(train_csv_path)
        mlflow.log_artifact(test_csv_path)

        # Training model setelah preprocessing
        from sklearn.linear_model import LinearRegression
        import numpy as np
        X_train_arr = np.array(X_train_final)
        X_test_arr = np.array(X_test_final)
        y_train_arr = np.array(y_train_clean)
        y_test_arr = np.array(y_test)

        model = LinearRegression()
        model.fit(X_train_arr, y_train_arr)
        y_pred = model.predict(X_test_arr)

        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(y_test_arr, y_pred)
        r2 = r2_score(y_test_arr, y_pred)

        # Log model ke MLflow
        mlflow.log_metric('mse', mse)
        mlflow.log_metric('r2', r2)
        mlflow.sklearn.log_model(model, 'model')

        return X_train_final, X_test_final, y_train_clean, y_test
