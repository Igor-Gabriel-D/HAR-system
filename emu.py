import time
import subprocess
import random  # Para simular os valores de x, y, z
import pandas as pd

# Função para enviar os dados via mosquitto_pub
def publish_data(broker, topic):
    try:
        df = pd.read_csv('jump_baseline.csv')
        df = df[['Ax1', 'Ay1', 'Az1']]

        # print(df)

        # print(df.shape[0])
        for i in range(df.shape[0]):
            # print(df.iloc[i])
            x = df.iloc[i]['Ax1']
            y = df.iloc[i]['Ay1']  # Valor aleatório entre -10 e 10
            z = df.iloc[i]['Az1']  # Valor aleatório entre -10 e 10

            message = f"{x},{y},{z}"


            subprocess.run(["mosquitto_pub", "-h", broker, "-t", topic, "-m", message])
            print(f"Enviado para {topic}: {message}")

            # Aguarda 200ms antes de enviar novamente
            time.sleep(0.2)
        # while True:
        #     # Simulando valores de aceleração para x, y, z (substitua por dados reais)
        #     x = random.uniform(-10, 10)  # Valor aleatório entre -10 e 10
        #     y = random.uniform(-10, 10)  # Valor aleatório entre -10 e 10
        #     z = random.uniform(-10, 10)  # Valor aleatório entre -10 e 10

        #     # Prepare a mensagem no formato desejado
        #     message = f"{x},{y},{z}"

        #     # Comando para publicar a mensagem com mosquitto_pub
        #     subprocess.run(["mosquitto_pub", "-h", broker, "-t", topic, "-m", message])
        #     print(f"Enviado para {topic}: {message}")

        #     # Aguarda 200ms antes de enviar novamente
        #     time.sleep(0.2)

    except KeyboardInterrupt:
        print("\nEncerrando envio de dados...")

if __name__ == "__main__":
    # Configuração do broker e tópico
    broker_address = "172.167.200.134"  # Endereço do broker MQTT
    topic_name = "/teste"  # Tópico para enviar os dados

    # Iniciar o envio de dados
    publish_data(broker_address, topic_name)
