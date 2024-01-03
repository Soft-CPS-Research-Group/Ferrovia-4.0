import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

STEP = 5000
DIRECTORY = "C:\\Users\\steam\\Desktop\\Dados Ferrovia\\"
df = pd.read_csv(DIRECTORY+'Contact_Forces_Rear_Bogie_Rear_Wheelset.csv')
num_lines = len(df)
data = np.array([])
#df_500 = df.iloc[:1000]
df_column =df['F_WSBB_L_x']

#dados_media = df.groupby(np.arange(len(df))//MEDIA_DE_DADOS).mean()
#print(dados_media)
fig, ax = plt.subplots()

# Initialize the plot with empty data
line, = ax.plot([], [])

# Set the plot limits and labels
ax.set_xlim(0, 0.005)
ax.set_xlabel('Frequência')
ax.set_ylabel('Amplitude')
ax.set_title('FFT')

for i in range(0, len(df_column), STEP):
    print("min->"+str(i))
    print("max->"+str(i+STEP))
    subset = df_column.iloc[i:i+STEP]
    fft_result = np.fft.fft(subset)
    freqs = np.fft.fftfreq(len(subset))
    #print(subset)
    line.set_xdata(np.abs(freqs))
    line.set_ydata(np.abs(fft_result))
    # Set the plot limits based on the current data
    ax.set_ylim(0, np.max(np.abs(fft_result)))
    # Draw the updated plot
    
    plt.draw()
    plt.pause(5)
plt.show()
time.sleep(10000)
#num_dados_media = num_lines - MEDIA_DE_DADOS + 1
#dt_total = dt * num_dados_media
sum_df = df.sum(axis=1)


fft_result = np.fft.fft(df_column)

# Calcule as frequências correspondentes aos resultados da FFT
freqs = np.fft.fftfreq(len(df_column))

plt.plot(freqs, np.abs(fft_result))
plt.xlim(-0.01,0.01)
plt.xlabel('Frequência')
plt.ylabel('Amplitude')
plt.show()
time.sleep(1000)
#for index, row in df.iterrows():
#    print(f"Row at index {index}:")
#    
#    print(row.values)
 #   time.sleep(1000)

def update_data():
    global data
    # Generate new random data
    new_data2 = np.random.randn(5000)
    new_data = np.mean(new_data2.reshape(-1, 100), axis=1)
    print(new_data)
    plt.plot(new_data)
    plt.show()
    # Append the new data to the existing data collection
    data = np.concatenate([data, new_data])
    
update_data()

def plot_fft():
    # Compute the FFT of the data
    fft_data = np.fft.fft(data)
    # Compute the frequency axis for the FFT data
    freq = np.fft.fftfreq(len(data))
    # Plot the FFT data
    plt.clf()
    plt.plot(freq, np.abs(fft_data))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("FFT of Incoming Data")
    plt.draw()
    plt.pause(0.001)

def timer():
    update_data()
    plot_fft()
    # Call the timer function again after 5 seconds
    plt.gcf().canvas.flush_events()
    plt.pause(5)
    timer()

# Start the timer function
timer()

