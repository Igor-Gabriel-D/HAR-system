# sudo apt -y install mosquitto-clients
# pip install scikit-learn
#  pip install pandas
# pip install tensorflow

from collections import Counter

import numpy as np

import subprocess
from sklearn.preprocessing import StandardScaler

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

from joblib import dump, load

from time import sleep

# jump = pd.read_csv('baseline_jump.csv')
# stand_baseline = pd.read_csv('stand_baseline.csv')

# stand_baseline = stand_baseline[['Ax1', 'Ay1', 'Az1']]

# signal_mpu = stand_baseline

# new_data = pd.DataFrame([{'Ax1': 5.5, 'Ay1': 9.5, 'Az1': 5.5}])

# # Adicionando a nova linha ao DataFrame
# signal_mpu = pd.concat([signal_mpu, new_data], ignore_index=True)
# signal_mpu = signal_mpu.drop(index=0).reset_index(drop=True)

# print(signal_mpu)

def step_features(df):

  df = df.iloc[:(df.shape[0] // 10)*10]

  # df = df.iloc[:25]

  transformed_df = None

  # aaaaaaa, faz um for para concatenar de 25 em 25 linhas, divide em sub dataframes e concatena tudo depois
  for i in range(df.shape[0] // 10):

    split_df = df.iloc[ 10*i:10*(i+1) ]

    column_copy_x = split_df[['Ax1']].copy()
    shifted_df = column_copy_x
    for j in range(2,11):
      shifted_df[f"Ax{j}"] = shifted_df[f"Ax{j-1}"].shift(periods=1)


    column_copy_y = split_df[['Ay1']].copy()
    shifted_df['Ay1'] = column_copy_y
    for j in range(2,11):
      shifted_df[f"Ay{j}"] = shifted_df[f"Ay{j-1}"].shift(periods=1)

    column_copy_z = df[['Az1']].copy()
    shifted_df['Az1'] = column_copy_z
    for j in range(2,11):
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


def monitor_mqtt_messages(broker, topic):
    
    try:

        jump = pd.read_csv('jump_baseline.csv')
        stand_baseline = pd.read_csv('stand_baseline.csv')

        stand_baseline = stand_baseline[['Ax1', 'Ay1', 'Az1']]

        signal_mpu = stand_baseline
        print(f"Conectando ao broker {broker} e monitorando o tópico '{topic}'...")
        
        # Executa o mosquitto_sub como subprocesso
        process = subprocess.Popen(
            ["mosquitto_sub", "-h", broker, "-t", topic],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        count = 0
        jump = False
        # Lê mensagens em tempo real
        mean = [0,0,0,0,0,0,0,0,0,0]  # Inicializa a lista com 10 zeros
        
        for line in iter(process.stdout.readline, ""):
            acc = line.strip().split(",")
            new_data = pd.DataFrame([{'Ax1': float(acc[1]), 'Ay1': float(acc[2]), 'Az1': float(acc[3])}])
            
            signal_mpu = pd.concat([signal_mpu, new_data], ignore_index=True).iloc[1:].reset_index(drop=True)
            pred = pipeline.predict(signal_mpu)
            res = np.argmax(pred[0])
            
            mean.pop(0)  # Remove o primeiro elemento
            mean.append(res)  # Adiciona o novo resultado no final
            
            most_common_value, _ = Counter(mean).most_common(1)[0]  # Encontra o valor mais frequente
            
            print(res)

            print(most_common_value)

            if most_common_value == 2:  # Se o valor mais frequente for 2, publica a mensagem
                subprocess.run(["mosquitto_pub", "-h", broker, "-t", "/mov", "-m", "2"])
                signal_mpu = stand_baseline.copy()
                mean = [0,0,0,0,0,0,0,0,0,0]
            
            if most_common_value == 3:  # Se o valor mais frequente for 2, publica a mensagem
                subprocess.run(["mosquitto_pub", "-h", broker, "-t", "/mov", "-m", "3"])
                signal_mpu = stand_baseline.copy()
                mean = [0,0,0,0,0,0,0,0,0,0]
            

            #count += 1



            # print(count)

            #signal_mpu = stand_baseline

    except KeyboardInterrupt:
        print("\nEncerrando monitoramento...")
        process.terminate()
    
    except Exception as e:
        print(f"Erro: {e}")

if __name__ == "__main__":
    # Configuração do broker e tópico
    broker_address = "172.167.200.134"  
    topic_name = "/teste"           

    monitor_mqtt_messages(broker_address, topic_name)





