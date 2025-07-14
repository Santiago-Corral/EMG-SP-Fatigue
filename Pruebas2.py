import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from scipy.fft import fft, fftfreq
from scipy import signal
from scipy.signal import savgol_filter

#%%
# Util functions

def plot_emg(time_data,emg_data,col_name):
    plt.figure(figsize=(10, 6))

    # Plot for raw data
    #plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
    plt.plot(time_data, emg_data, label="EMG Data", color='blue')
    plt.xlabel("Time (s)")
    plt.ylabel("EMG (V)")
    plt.title(str(col_name))
    plt.grid(True)

    # Display the plot

    #plt.tight_layout()
    plt.show()


def plot_emg_wlabels(x, y, name=None, label='Frequencies', gt_time=None):
    plt.figure(figsize=(12, 6))
    plt.plot(x, y)
    plt.title(str(label), fontsize=16) 
    plt.xlabel('Time (s)', fontsize=14)  
    plt.ylabel('Frequency (Hz)', fontsize=14)  
    plt.grid(True)
    plt.axvline(x=gt_time[0], color='black', linestyle='--', label='Non-Fatigue')
    plt.axvline(x=gt_time[1], color='black', linestyle='--', label='Transition-to-Fatigue')
    
    if name is not None:
        plt.savefig(name)
        print("saved at", name)
    plt.close()
    

def mean_frequency(power_spectrum, frequencies):
    """
    Calculate the mean frequency from the power spectrum.
    Args:
    - power_spectrum: Power spectrum from FFT.
    - frequencies: Corresponding frequency bins.
    
    Returns:
    - Mean frequency.
    """

    if len(power_spectrum) == 0 or np.sum(power_spectrum) == 0:
        print("-------- power spectrum is empty --------")
        return None  
    

    weighted_sum = np.sum(frequencies * power_spectrum)
    total_power = np.sum(power_spectrum)
    
    if total_power == 0:
        print( "------------ Total Power is zero --------- ")
        return None
    
    mean_freq = weighted_sum / total_power
    
    return mean_freq


def median_frequency(power_spectrum, frequencies):
    """
    Calculate the median frequency from the power spectrum.
    Args:
    - power_spectrum: Power spectrum from FFT.
    - frequencies: Corresponding frequency bins.
    
    Returns:
    - Median frequency.
    """
    cumulative_power = np.cumsum(power_spectrum)
    total_power = cumulative_power[-1] #last element retrives the total power
    
    # Find the frequency where the cumulative power reaches 50% of the total power
    median_idx = np.where(cumulative_power >= (total_power / 2))[0][0]

    return frequencies[median_idx]


def calculate_median_frequencies(signal, time, sampling_rate, window_size_seconds, step_size_seconds):
    """
    Calculate the median frequency using a sliding window approach.
    Args:
    - signal: The raw signal values.
    - sampling_rate: The sampling rate of the signal.
    - window_size_seconds: Duration of each window in seconds.
    - step_size_seconds: Step size for sliding the window in seconds.
    
    Returns:
    - List of median frequencies for each window.
    """

    window_size = int(window_size_seconds * sampling_rate)  # Number of samples in one window
    step_size = int(step_size_seconds * sampling_rate)      # Number of samples to slide the window
    num_windows = int((len(signal) - window_size) // step_size + 1)
    
    median_frequencies = []
    mean_frequencies = []
    window_time_center = []

    for i in range(num_windows):

        # Get the current window
        start_idx = i * step_size
        end_idx = start_idx + window_size
        window_signal = signal[start_idx:end_idx]
        window_time = time[start_idx:end_idx]

        # Calculate the center time for the current window
        center_time = window_time[int(len(window_time) // 2)]
        window_time_center.append(center_time)

        signal_mean = np.mean(signal)
        #plot(window_time,window_signal, name=filename, label='raw sEMG windows')

        # Perform FFT on the windowed signal
        fft_result = fft(window_signal-signal_mean)
        frequencies = fftfreq(len(window_signal), d=1/sampling_rate)
        power_spectrum = np.abs(fft_result)**2  # Power spectrum

        # Only consider positive frequencies (first half of FFT result)
        positive_frequencies = frequencies[:len(frequencies)//2]
        positive_power_spectrum = power_spectrum[:len(power_spectrum)//2]  

        # Calculate median frequency for this window
        med_freq = median_frequency(positive_power_spectrum, positive_frequencies)
        mean_freq = mean_frequency(positive_power_spectrum, positive_frequencies)
        median_frequencies.append(med_freq)
        mean_frequencies.append(mean_freq)


    return window_time_center, median_frequencies, mean_frequencies


# Function to normalize time by maximum duration
def normalize_time_to_max(raw_time, max_time):
    if raw_time[-1] < max_time:
        # Pad with NaNs to match the max_time
        padding = np.nan * np.ones(int(max_time - len(raw_time)))
        return np.concatenate((raw_time, padding))
    else:
        # Truncate if longer
        return raw_time
    

def plot_overlapping_data(time_data, muscle_data, metric, muscle_name):
    plt.figure(figsize=(20, 14))

    cmap = plt.get_cmap('tab20')  
    colors = [cmap(i) for i in range(13)]  

    line_styles = ['-', '--']  
    line_styles = line_styles * 7  # Repeat to match the number of subjects
    
    # Loop through subjects
    for subject_number in range(1, 14):  # Loop through all subjects (1 to 13)
        subject_key = f'subject_{subject_number}'
        
        for trial_index in range(len(time_data[subject_key])):  # Loop through trials for each subject
                    time = time_data[subject_key][trial_index]
                    data = muscle_data[subject_key][trial_index]
                    if len(time) == len(data):  # Ensure that time and data have matching lengths

                        smoothed_data = savgol_filter(data, window_length=8, polyorder=3)  # Adjust window_length and polyorder for best results

                        # Plot smoothed trend
                        plt.plot(time, smoothed_data, label=f'Subject {subject_number} - {muscle_name}', 
                                 color=colors[subject_number - 1], linestyle=line_styles[subject_number % len(line_styles)])
                    else:
                        print(f"Time and data lengths mismatch for subject {subject_number}, {metric}")

    # Set plot titles and labels once after all subjects are plotted
    #plt.title(f'{muscle_name} ', fontsize=18)
    plt.xlabel('Time (s)', fontsize=18)
    plt.ylabel(f'{metric} (Hz)', fontsize=18)
    plt.legend(loc='best', fontsize=14)  # Move legend outside the plot for clarity
    plt.grid(True)

    # Save the plot
    name = f"{metric}_{muscle_name.replace(' ', '_')}.png"
    filename = os.path.join(dir, name)
    plt.savefig(filename)
    #plt.show()
    plt.close()
    print(f"Saved plot at {filename}")

    #associate muscle with trial
def get_prime_mover(trial):
    mapping = {1:1,2:0, # anterior
               3:3,4:2, # posterior
               5:0,6:1, # biceps
               7:2,8:3, # medius
               9:1,10:0, # anterior
               11:3,12:2} # posterior
    return mapping.get(trial,None)

# Trial-to-muscle correspondence dictionary
trial_to_muscle = {
    1: "R DELTOID ANTERIOR",
    2: "L DELTOID ANTERIOR",
    3: "R DELTOID POSTERIOR",
    4: "L DELTOID POSTERIOR",
    5: "R BICEPS BRACHII",
    6: "L BICEPS BRACHII",
    7: "R DELTOID MEDIUS",
    8: "L DELTOID MEDIUS",
    9: "R DELTOID ANTERIOR C",
    10: "L DELTOID ANTERIOR C",
    11: "R DELTOID POSTERIOR C",
    12: "L DELTOID POSTERIOR C",
}

#%%
dir= r"C:\Users\santi\Desktop\Facu\APS\TP Final\EMG-Self-Perceived\sEMG_data"
gt_dir =  r"C:\Users\santi\Desktop\Facu\APS\TP Final\EMG-Self-Perceived\self_perceived_fatigue_index"


subject_numbers = range(1, 2)  # Adjust the range as needed

not_filtered = True
X_subjects = []
true_labels_subjects = []
time= []
EMG_index = [1,3,5,7]
time_index = [0,2,4,6]
sample_freq = 1259 #Hz


# Initialize a dictionary to store the data for each muscle
# Initialize a dictionary to store the MNF/MDF data for each muscle, subject, and trial
muscle_data_MNF = {f'muscle_{i}': {f'subject_{j}': [] for j in range(1, 14)} for i in range(1, 14)}
muscle_data_MDF = {f'muscle_{i}': {f'subject_{j}': [] for j in range(1, 14)} for i in range(1, 14)}
time_dic = {f'muscle_{i}': {f'subject_{j}': [] for j in range(1, 14)} for i in range(1, 14)}

# Step 1: Find the maximum time across all subjects and trials
max_time = 0

# Loop through each subject
for subject_number in subject_numbers:
    emg_directory = os.path.join(dir, f'Subject_{subject_number}')
    #GT_directory = os.path.join(dir, f'Subject_{subject_number}')
    
    emg_files = sorted([f for f in os.listdir(emg_directory) if f.endswith(".csv")], key=lambda x: int(re.findall(r'\d+', x)[-1]))
    #GT_files = sorted([f for f in os.listdir(GT_directory) if f.endswith(".csv")], key=lambda x: int(re.findall(r'\d+', x)[-1]))

    data_subject = []  # List to store data for current subject
    labels_subject = []  # List to store true labels for current subject

    for emg_file in zip(emg_files):
        emg_path = os.path.join(emg_directory, emg_file[0])

        file = pd.read_csv(emg_path, delimiter=',', header=0)
        file_values = file.values
        columns_names = file.columns
        trial = int(re.findall(r'\d+', emg_file[0])[-1])
    
        # Loop through EMG and time indices
        index = get_prime_mover(trial)
        print("trial", trial, index)
        

        emg_data = file_values[:, EMG_index[index]]
        time_data = file_values[:, time_index[index]]
        col_name = columns_names[EMG_index[index]]
        #plot_emg(time_data, emg_data,col_name)

        if not_filtered:
            fc1, fc2 = 20, 450
            b, a = signal.butter(4, [fc1, fc2], btype='bandpass', fs=sample_freq)
            filtered_emg = signal.filtfilt(b, a, emg_data)
            emg_data = filtered_emg


        # Define window and step size
        window_size_seconds = 4  # 4-second window
        step_size_seconds = 2    # 50% overlap
        #Overlapping windows help reduce abrupt changes between segments and provide a more continuous analysis of the signal.

        # Calculate median frequencies
        window_time_centers, median_frequencies, mean_frequencies = calculate_median_frequencies(emg_data,time_data, sample_freq, window_size_seconds, step_size_seconds )

        ## ---- Single emg  plots --- ##
        #plot_emg(window_time_centers,median_frequencies,("MDF" + col_name))
        #plot_emg(window_time_centers,mean_frequencies,("MNF" + col_name))

        max_time = max(max_time, window_time_centers[-1])  # Update the maximum time
        

        # Store the data by muscle

        muscle_data_MNF[f'muscle_{trial}'][f'subject_{subject_number}'].append(mean_frequencies)
        muscle_data_MDF[f'muscle_{trial}'][f'subject_{subject_number}'].append(median_frequencies)
        time_dic[f'muscle_{trial}'][f'subject_{subject_number}'].append(window_time_centers)

for trial in range(1, 13):
    # Get the corresponding muscle name from the trial index
    muscle_name = trial_to_muscle.get(trial)
    print(f"Plotting data for {muscle_name}")

    # Check if there is any data to plot
    if time_dic[f'muscle_{trial}'] and muscle_data_MNF[f'muscle_{trial}']:
        # Plot MNF
        plot_overlapping_data(time_dic[f'muscle_{trial}'], muscle_data_MNF[f'muscle_{trial}'], 
                              'Mean Frequency', f'{muscle_name}')

    if time_dic[f'muscle_{trial}'] and muscle_data_MDF[f'muscle_{trial}']:
        # Plot MDF
        plot_overlapping_data(time_dic[f'muscle_{trial}'], muscle_data_MDF[f'muscle_{trial}'], 
                              'Median Frequency', f'{muscle_name}')
    else:
        print(f"No data to plot for muscle {muscle_name}")

#%%
## Frequency analysis  - MDF and MNF
# PLOT a graph with the MDF values of 1 cycle
# velocity was 30 bpm, meaning that there was a beat every 2 seconds. 
#therefore, the cycle was 2 seconds and the window size should also be 

from scipy import signal

base_directory = r'[insert path to your folder]'

EMG_index = [1,3,5,7]
time_index = [0,2,4,6]
sample_freq = 1259 #Hz

for subject_number in range(1, 14): 

  path_directory = os.path.join(base_directory, 'sEMG_data', f'subject_{subject_number}')
  label_directory = os.path.join(base_directory, 'self_perceived_fatigue_index', f'subject_{subject_number}')
  global out_directory
  out_directory = os.path.join(base_directory, '','frequency_analysis', 'per_trial',f'subject_{subject_number}')

  isExist = os.path.exists(out_directory)
  if not isExist:


    os.makedirs(out_directory)


  files = [file_name for file_name in os.listdir(path_directory) if file_name.endswith("csv")]
  files = sorted(files, key=lambda x: int(re.findall(r'\d+', x)[-1]))

  label_files = [file_name for file_name in os.listdir(label_directory) if file_name.endswith("csv")]
  label_files = sorted(label_files, key=lambda x: int(re.findall(r'\d+', x)[-1]))
  
  for file_raw, file_down in zip(files, label_files):
        # Read the CSV file
        
        raw_files_path = os.path.join(path_directory, file_raw)
        label_files_path = os.path.join(label_directory, file_down)

        trial = re.findall(r'\d+', file_raw)
        print("file", file_raw, "trial", int(trial[0]))
        index = get_prime_mover(int(trial[0]))


        raw = pd.read_csv(raw_files_path, delimiter=',', header=0)
        file_values = raw.values
        raw_time =  file_values[:, time_index[index]]
        raw_val =  file_values[:, EMG_index[index]]
        if not_filtered:
          fc1, fc2 = 20, 450
          b, a = signal.butter(4, [fc1, fc2], btype='bandpass', fs=sample_freq)
          filtered_emg = signal.filtfilt(b, a, raw_val)
          raw_val = filtered_emg

        label = pd.read_csv(label_files_path, delimiter=',', header=0)
        label_time = label.iloc[:, 0]  
        label_values = label.iloc[:, 1]

        raw_values = np.array(raw_val)
        if np.isnan(raw_val).any():
          print("NaN values found in the signal. Replacing NaN with zeros.")
    
        if np.isinf(raw_val).any():
            print("Inf values found in the signal. Replacing Inf with zeros.")

        raw_values = np.where(np.isnan(raw_val) | np.isinf(raw_val), 0, raw_val)

      # ----- Segment 0: From the beginning to the last 0 in label_values
        idx_last_0 = label_values[label_values == 0].index[-1]
        segment_0_label_time = label_time[idx_last_0]
        # Find the index in raw_time that corresponds to the nearest time to segment_1_label_time
        nearest_idx = (np.abs(raw_time - segment_0_label_time)).argmin()
        # Nearest time and values in raw_time
        nearest_time = raw_time[nearest_idx]

        # ------- Segment 1: From the the fisrt 1 to the last 1 in label_values
        idx_last_1 = label_values[label_values == 1].index[-1]
        segment_1_label_time = label_time[idx_last_1]
        nearest_idx_1 = (np.abs(raw_time - segment_1_label_time)).argmin()
        nearest_time_1 = raw_time[nearest_idx_1]

        gt_times = [nearest_time, nearest_time_1]
  
        # Define window and step size FOR MDF AND MNF ANALYSIS
        window_size_seconds = 4  # 4-second window
        step_size_seconds = 2    # 4-second step (50% overlap) 
        #Overlapping windows help reduce abrupt changes between segments and provide a more continuous analysis of the signal.

        # Calculate median frequencies
        window_time_centers, median_frequencies,mean_frequencies = calculate_median_frequencies(raw_values,raw_time, sample_freq, window_size_seconds, step_size_seconds )

        print(f"Median Frequencies for {file_raw}: {median_frequencies}")
        freq_MDF_name = os.path.join(out_directory,f'MDF_{(file_raw.replace(".csv",".png"))}')
        #id = np.arange(0,len(median_frequencies),1)
        plot_emg_wlabels(window_time_centers,median_frequencies,name=freq_MDF_name, label="median frequencies", gt_time=gt_times)


        #n = np.arange(0,len(mean_frequencies),1)
        freq_name = os.path.join(out_directory,f'MNF_{(file_raw.replace(".csv",".png"))}')
        plot_emg_wlabels(window_time_centers,mean_frequencies, name = freq_name, label='mean frequencies', gt_time=gt_times)
        print ("---")
        
#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# --- Configuración ---
emg_dir = r"C:\Users\santi\Desktop\Facu\APS\TP Final\Definitivos\EMG-Self-Perceived\sEMG_data"
subject_number = 1
trial_number = 1
sample_freq = 1259  # Hz
EMG_index = [1, 3, 5, 7]
time_index = [0, 2, 4, 6]

# --- Leer archivo correspondiente ---
file_name = f"trial_{trial_number}.csv"
file_path = os.path.join(emg_dir, f"Subject_{subject_number}", file_name)

df = pd.read_csv(file_path)
data = df.values
columns = df.columns

# --- Filtro Butterworth ---
fc1, fc2 = 20, 450
b, a = signal.butter(4, [fc1, fc2], btype='bandpass', fs=sample_freq)

# --- Plotear los 4 canales EMG ---
plt.figure(figsize=(12, 8))
for i, idx in enumerate(EMG_index):
    emg_raw = data[:, idx]
    emg_filtered = signal.filtfilt(b, a, emg_raw)
    time = data[:, time_index[i]]

    plt.subplot(4, 1, i+1)
    plt.plot(time, emg_filtered, label=f"{columns[idx]}", color='blue')
    plt.title(f"Sujeto {subject_number} - Actividad {trial_number} - Canal {columns[idx]}")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("EMG (V)")
    plt.grid(True)
    plt.tight_layout()

plt.suptitle(f"Sujeto {subject_number} - Actividad {trial_number} - Señales EMG filtradas", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

#%% CHAT FIGURA 8

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# --- Parámetros básicos ---
subject_number = 2
trial_number = 2
sample_freq = 1259
EMG_index = [1, 3, 5, 7]
time_index = [0, 2, 4, 6]

# --- Rutas ---
emg_base = r"C:\Users\santi\Desktop\Facu\APS\TP Final\Definitivos\EMG-Self-Perceived\sEMG_data"
gt_base  = r"C:\Users\santi\Desktop\Facu\APS\TP Final\Definitivos\EMG-Self-Perceived\self_perceived_fatigue_index"

emg_path = os.path.join(emg_base, f"subject_{subject_number}", f"trial_{trial_number}.csv")
gt_path  = os.path.join(gt_base,  f"subject_{subject_number}", f"trial_{trial_number}.csv")

# --- Leer señales EMG ---
df = pd.read_csv(emg_path)
data = df.values
columns = df.columns

# --- Leer etiquetas de fatiga subjetiva ---
labels_df = pd.read_csv(gt_path)
label_times = labels_df.iloc[:, 0].values
label_values = labels_df.iloc[:, 1].values

print("Niveles presentes en etiquetas:", np.unique(label_values))

# Buscar los tiempos de transición
t1 = None
t2 = None
try:
    t1 = label_times[np.where(label_values == 1)[0][0]]
except IndexError:
    print("⚠️ No se encontró fatiga nivel 1")

try:
    t2 = label_times[np.where(label_values == 2)[0][0]]
except IndexError:
    print("⚠️ No se encontró fatiga nivel 2")

# --- Filtro Butterworth 20–450 Hz ---
b, a = signal.butter(4, [20, 450], btype='bandpass', fs=sample_freq)

# --- Graficar los 4 canales EMG ---
plt.figure(figsize=(12, 10))
for i in range(4):
    emg_raw = data[:, EMG_index[i]]
    time = data[:, time_index[i]]
    emg_filtered = signal.filtfilt(b, a, emg_raw)

    plt.subplot(4, 1, i + 1)
    plt.plot(time, emg_filtered, label=f"{columns[EMG_index[i]]}", color='blue')
    plt.ylabel("EMG (V)")
    plt.title(f"Canal: {columns[EMG_index[i]]}")
    plt.grid(True)
    
    if t1 is not None:
        plt.axvline(x=t1, linestyle='--', color='black', label='Fatiga nivel 1')
    if t2 is not None:
        plt.axvline(x=t2, linestyle='--', color='red', label='Fatiga nivel 2')

    if i == 0:
        plt.legend()

plt.xlabel("Tiempo (s)")
plt.suptitle(f"Sujeto {subject_number} - Ensayo {trial_number} - Señales EMG (filtradas)", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

#%% CHAT GPT SEÑAL CRUDA
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# --- Parámetros básicos ---
subject_number = 2
trial_number = 2
sample_freq = 1259  # Hz
EMG_index = [1, 3, 5, 7]  # columnas de EMG en CSV
time_index = [0, 2, 4, 6]  # columnas de tiempo para cada canal

# --- Rutas ---
emg_base = r"C:\Users\santi\Desktop\Facu\APS\TP Final\EMG-Self-Perceived\sEMG_data"
gt_base  = r"C:\Users\santi\Desktop\Facu\APS\TP Final\EMG-Self-Perceived\self_perceived_fatigue_index"

emg_path = os.path.join(emg_base, f"subject_{subject_number}", f"trial_{trial_number}.csv")
gt_path  = os.path.join(gt_base,  f"subject_{subject_number}", f"trial_{trial_number}.csv")

# --- Leer señales EMG ---
df = pd.read_csv(emg_path)
data = df.values
columns = df.columns

# --- Leer etiquetas fatiga subjetiva ---
labels_df = pd.read_csv(gt_path)
label_times = labels_df.iloc[:, 0].values
label_values = labels_df.iloc[:, 1].values

print("Niveles presentes en etiquetas:", np.unique(label_values))

# Buscar tiempos de transición fatiga nivel 1 y 2 (si existen)
t1 = None
t2 = None
try:
    t1 = label_times[np.where(label_values == 1)[0][0]]
except IndexError:
    print("⚠️ No se encontró fatiga nivel 1")
try:
    t2 = label_times[np.where(label_values == 2)[0][0]]
except IndexError:
    print("⚠️ No se encontró fatiga nivel 2")

# --- Filtro Butterworth 20–450 Hz ---
b, a = signal.butter(4, [20, 450], btype='bandpass', fs=sample_freq)

# --- PLOT 1: Comparación crudo vs filtrado en canal 1 (primer canal EMG) ---
channel_i = 0
emg_raw = data[:, EMG_index[channel_i]]
time_raw = data[:, time_index[channel_i]]
emg_filtered = signal.filtfilt(b, a, emg_raw)

# Elegir segmento para zoom (ejemplo 2 segundos)
start_idx = 5000
end_idx = start_idx + 2 * sample_freq  # 2 segundos
t_zoom = time_raw[start_idx:end_idx]

plt.figure(figsize=(12, 5))
plt.plot(t_zoom, emg_raw[start_idx:end_idx], label='Raw EMG', alpha=0.6)
plt.plot(t_zoom, emg_filtered[start_idx:end_idx], label='Filtered EMG', linewidth=1.5)
plt.title(f'Sujeto {subject_number} - Ensayo {trial_number} - Canal {columns[EMG_index[channel_i]]} (Zoom)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- PLOT 2: Señal completa filtrada con líneas de fatiga ---
plt.figure(figsize=(15, 5))
plt.plot(time_raw, emg_filtered, color='navy', label='Filtered EMG')
    
if t1 is not None:
    plt.axvline(x=t1, color='black', linestyle='--', alpha=0.7, label='Fatiga nivel 1')
if t2 is not None:
    plt.axvline(x=t2, color='red', linestyle='--', alpha=0.7, label='Fatiga nivel 2')

plt.title(f'Sujeto {subject_number} - Ensayo {trial_number} - Canal {columns[EMG_index[channel_i]]} (Señal completa)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


