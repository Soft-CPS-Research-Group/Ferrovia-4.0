import pandas as pd
import time
import os
import json
import paho.mqtt.client as mqtt
import multiprocessing

# Define the MQTT broker address and port
mqtt_broker_address = 'localhost'
mqtt_broker_port = 1883

# Define the topic to publish to


# Define the number of rows to publish for each dataframe
num_rows_per_df = 10

# Define the delay in seconds between each batch of 100 rows
delay_seconds = 0.3



# Define Directory to get csv names
csv_dir = 'C:\\Users\\steam\\Desktop\\Bolsa\\Ferrovia\\data'


# Define the list of dataframe file names
df_file_names = [csv_dir+"\\"+f for f in os.listdir(csv_dir) if f.endswith('.csv')]


def main_fun(df_file):
    def on_connect(client, userdata, flags, rc):
        print(f"Connected to MQTT broker with result code {rc}")

    def on_disconnect(client, userdata, rc):
        print(f"Disconnected from MQTT broker with result code {rc}")

    # Set up the MQTT client
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.connect(mqtt_broker_address, mqtt_broker_port)
    while True:
        # Read the dataframe from the CSV file
        df = pd.read_csv(df_file)
        print(df.columns)
        # Get the total number of rows in the dataframe
        num_rows = df.shape[0]

        # Get the column names from the dataframe
        cols = df.columns.tolist()

        # Loop through the rows in batches of 100
        for i in range(0, num_rows, num_rows_per_df):
            # Get the next batch of rows from the dataframe
            batch = df[i:i+num_rows_per_df]
            
            # Convert the batch of rows to a list of dictionaries
            rows = batch.to_dict(orient='records')
            #print(rows)
            # Prepend the column names to the first row of the batch
            rows_with_header = [cols] + [[row[col] for col in cols] for row in rows]

            # Publish the list of rows with column names as a JSON string to the MQTT topic
            message = json.dumps(rows_with_header)
            mqtt_topic = df_file.replace(".csv","")
            mqtt_topic = mqtt_topic.replace(csv_dir+"\\","")
            client.publish(mqtt_topic, payload=message)

            # Wait for the specified delay between publishing each batch of rows
            time.sleep(delay_seconds)

if __name__ == '__main__':
    p = None
    for x in df_file_names:
        print(x)
        p = multiprocessing.Process(target=main_fun, args=(x,))
        p.start()
    p.join()
    

