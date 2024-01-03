import pandas as pd
import time
import paho.mqtt.client as mqtt
import pandas as pd

mqtt_broker_address = 'localhost'
mqtt_broker_port = 1883
chunk_size = 10
delay_seconds = 0.001
csv_dir = 'C:\\Users\\steam\\Desktop\\Bolsa\\Ferrovia\\AI Ferrovia'
TOPIC = "main_file"




def main_fun():
    def on_connect(client, userdata, flags, rc):
        print(f"Connected to MQTT broker with result code {rc}")

    def on_disconnect(client, userdata, rc):
        print(f"Disconnected from MQTT broker with result code {rc}")

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.connect(mqtt_broker_address, mqtt_broker_port)
    df = pd.read_excel(csv_dir+"\\"+'_aux.xlsx')
    print(df.columns)
    while True:
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            payload = chunk.to_json(orient="records")
            client.publish(TOPIC, payload)
            time.sleep(delay_seconds)
        payload = "finish"
        client.publish(TOPIC, payload)

if __name__ == '__main__':
    main_fun()
    

