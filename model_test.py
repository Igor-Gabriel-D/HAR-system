# sudo apt -y install mosquitto-clients
# pip install scikit-learn
#  pip install pandas
# pip install tensorflow

import numpy as np

import subprocess
from sklearn.preprocessing import StandardScaler

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

from joblib import dump, load

df = pd.read_csv('jump.csv')
print(df)



# new_data = pd.DataFrame([{'Ax1': 5.5, 'Ay1': 9.5, 'Az1': 5.5}])

# # Adicionando a nova linha ao DataFrame
# signal_mpu = pd.concat([signal_mpu, new_data], ignore_index=True)
# signal_mpu = signal_mpu.drop(index=0).reset_index(drop=True)

# print(signal_mpu)

def step_features(df):

  df = df.iloc[:(df.shape[0] // 25)*25]

  # df = df.iloc[:25]

  transformed_df = None

  # aaaaaaa, faz um for para concatenar de 25 em 25 linhas, divide em sub dataframes e concatena tudo depois
  for i in range(df.shape[0] // 25):

    split_df = df.iloc[ 25*i:25*(i+1) ]

    column_copy_x = split_df[['Ax1']].copy()
    shifted_df = column_copy_x
    for j in range(2,26):
      shifted_df[f"Ax{j}"] = shifted_df[f"Ax{j-1}"].shift(periods=1)


    column_copy_y = split_df[['Ay1']].copy()
    shifted_df['Ay1'] = column_copy_y
    for j in range(2,26):
      shifted_df[f"Ay{j}"] = shifted_df[f"Ay{j-1}"].shift(periods=1)

    column_copy_z = df[['Az1']].copy()
    shifted_df['Az1'] = column_copy_z
    for j in range(2,26):
      shifted_df[f"Az{j}"] = shifted_df[f"Az{j-1}"].shift(periods=1)


    shifted_df = shifted_df.dropna(axis=0)

    if i == 0:
      transformed_df = shifted_df
    else:
      transformed_df = pd.concat([transformed_df, shifted_df], ignore_index=True)

  return transformed_df

class StepFeaturesTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):

        if not isinstance(X, pd.DataFrame):
            raise ValueError("O input deve ser um DataFrame.")
        return step_features(X)

pipeline = load('pipeline.joblib')


preds = pipeline.predict(df)

print(preds)


