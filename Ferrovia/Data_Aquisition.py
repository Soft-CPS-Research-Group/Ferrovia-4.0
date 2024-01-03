import pandas as pd
import json
import paho.mqtt.client as mqtt

# Define the MQTT broker address and port
mqtt_broker_address = 'localhost'
mqtt_broker_port = 1883

# Define the topic to subscribe to
mqtt_topic = 'example_topic'

# Set up the MQTT client
client = mqtt.Client()

# Define a callback function to handle incoming messages
def on_message(client, userdata, message):
    # Convert the message payload from bytes to string
    message_str = message.payload.decode('utf-8')

    # Parse the JSON string into a list of rows with column names
    rows_with_header = json.loads(message_str)

    # Convert the list of rows with column names to a pandas dataframe
    df = pd.DataFrame(rows_with_header[1:], columns=rows_with_header[0])
    cols = df.columns.tolist()
    # Print the pandas dataframe
    print(cols)

# Set the on_message callback function for the MQTT client
client.on_message = on_message

# Connect to the MQTT broker and subscribe to the topic
client.connect(mqtt_broker_address, mqtt_broker_port)
client.subscribe(mqtt_topic)

# Start the MQTT client loop to receive incoming messages
client.loop_forever()