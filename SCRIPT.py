import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal import stft
import scipy as sp

# %% FUNCIONES

# //// Plot los 4 canales dado un trial y sujeto //// 
def plot_emg(data_dict, subject = 1, trial = 1):
    subjectt = f'subject_{subject}'
    triall = f'trial_{trial}'
    
    # Acceder a los datos
    emg = data_dict[subjectt][triall]['emg']     # matriz Nx4
    time = data_dict[subjectt][triall]['time']   # matriz Nx4
    labels_time = data_dict[subjectt][triall]['label_times']
    labels = data_dict[subjectt][triall]['labels']
    
    # Buscar los tiempos de transición
    t1 = None
    t2 = None

    t1 = labels_time[np.where(labels == 1)[0][0]]
    t2 = labels_time[np.where(labels == 2)[0][0]]
    
    #PLOT
    plt.figure(figsize=(12, 10))
    plt.title(f"Sujeto {subject} Ejercicio {trial}")
    
    for i in range(4):
        plt.subplot(4, 1, i + 1)
        plt.plot(time[:, i], emg[:, i], label=f'Canal {i + 1}')
        plt.axvline(x=t1, linestyle='--', color='black', label='Fatiga nivel 1' if i == 0 else "")
        plt.axvline(x=t2, linestyle='--', color='red', label='Fatiga nivel 2' if i == 0 else "")
        plt.ylabel("EMG (V)")
        plt.grid(True)
        plt.legend(loc='upper right')

    plt.xlabel("Tiempo (s)")
    plt.tight_layout()
    plt.show()

# //// Plot canal ppal dado un trial para todos los sujetos ////     
def plot_emg_all_canal(data_dict, trial_num, canal):
    
    plt.figure()  # Altura grande para 13 subplots

    for i in range(1, 7):  # sujetos
        subject = f'subject_{i}'
        trial = f'trial_{trial_num}'

        plt.subplot(13, 1, i)  # 13 filas, 1 columna, subplot i
       
        emg = data_dict[subject][trial]['emg']
        time = data_dict[subject][trial]['time']
            
        plt.plot(time[:, canal], emg[:, canal], label=subject)
        plt.ylabel("EMG (V)")
        plt.grid(True)
        plt.legend(fontsize=8, loc='upper right')
    

    plt.suptitle(f"Trial {trial_num} - Canal {canal + 1}", fontsize=16, y=0.92)
    plt.xlabel("Tiempo (s)")
    plt.tight_layout()
    plt.show()

# //// Plot canal ppal dado un trial para todos los sujetos ////     
def plot_emg_all(data_dict, trial_num):
    
    plt.figure()  # Altura grande para 13 subplots

    for i in range(1, 7):  # sujetos
        subject = f'subject_{i}'
        trial = f'trial_{trial_num}'
        

        plt.subplot(7, 1, i)  # 13 filas, 1 columna, subplot i
       
        #Carga de datos
        emg = data_dict[subject][trial]['emg']
        time = data_dict[subject][trial]['time']
        labels_time = data_dict[subject][trial]['label_times']
        labels = data_dict[subject][trial]['labels']
        
        # Buscar los tiempos de transición
        t1 = None
        t2 = None

        t1 = labels_time[np.where(labels == 1)[0][0]]
        t2 = labels_time[np.where(labels == 2)[0][0]]
        
        #Plot
        plt.plot(time[:], emg[:], label=subject)
        plt.axvline(x=t1, linestyle='--', color='black', label='Fatiga nivel 1' if i == 1 else None)
        plt.axvline(x=t2, linestyle='--', color='red', label='Fatiga nivel 2' if i == 1 else None)
        plt.ylabel("EMG (V)")
        plt.grid(True)
        plt.legend(loc='upper right')
    

    plt.suptitle(f"Trial {trial_num}", fontsize=16, y=0.92)
    plt.xlabel("Tiempo (s)")
    plt.show()    

# //// Mapa con el canal del musculo ppal. para cada trial //// 
def get_agonista(trial):
    mapping = {
        1: 1,  2: 0,     # Deltoides anterior
        3: 3,  4: 2,     # Deltoides posterior
        5: 0,  6: 1,     # Biceps Brachii
        7: 2,  8: 3,     # Deltoide Medio
        9: 1, 10: 0,     # Deltoide anterior (complejo)
        11:3, 12:2       # Deltoide posterior (complejo)
    }
    return mapping.get(trial, None)

# //// Esta creo que se puede volar ////
def plot_fft_all_subjects(data_dict, fs, trial_number):
    
    plt.figure(figsize=(12, 6))
    trial_key = f"trial_{trial_number}"
    
    for subj in range(1, 13):  # 12 sujetos
        subject_key = f"subject_{subj}"
        emg = data_dict[subject_key][trial_key]['emg']
        N = len(emg)
        freq = fftfreq(N, 1/fs)
        fft_vals = np.abs(fft(emg)) / N  # Normalizamos por N
            
        # Solo frecuencias positivas
        mask = freq >= 0
        freq = freq[mask]
        fft_vals = fft_vals[mask]
            
        plt.plot(freq, fft_vals, label=f"Sujeto {subj}", alpha=0.6)
    
    plt.title(f"Espectro de frecuencia - Trial {trial_number}")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud (uV)")
    plt.xlim(0, 500)  # Frecuencia útil del EMG
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.show()
    
# //// Calcula la frecuencia media del espectro  ////
def mean_frequency(espectro_pot, frequencies):
    
    suma_ponderada = np.sum(frequencies * espectro_pot)
    potencia_total = np.sum(espectro_pot)
    
    return suma_ponderada / potencia_total #Frecuencia media

# //// Calcula la frecuencia mediana del espectro  ////
def median_frequency(espectro_pot, frequencies):
    
    pot_acumulada = np.cumsum(espectro_pot)
    # pot_acumulada = [P0, P0 + P1, P0 + P1 + P2, ... , P0 + ... + Pn]
    
    pot_total = np.sum(espectro_pot) #Sumatoria del espectro de potencias
    
    #La funcion where devuelve los indices en los que se cumple la condición
    #Tomamos el primer indice del array ([0][0], este es exactamente el pto. medio)
    median_idx = np.where(pot_acumulada >= (pot_total / 2))[0][0] 
    
    return frequencies[median_idx] 

# //// Metodo sliding win //// Devuelve MDF MNF y bins de frecuencias
def slidingwin_fft_mnf_mdf(signal, time, fs, window_sec=4, step_sec=2):
    
    #Analizamos la fft en ventanas de 4sec con paso de 2sec (solapamiento del 50%)
    
    window_size = int(window_sec * fs) #en muestras
    step_size = int(step_sec * fs) #en muestras
    num_windows = (len(signal) - window_size) // step_size + 1

    mnf_list = []
    mdf_list = []
    time_centers = []

    for i in range(num_windows): #Recorremos todas las ventanas
        start = i * step_size #Inicio de la ventana 
        end = start + window_size #Fin de la ventana
        segment = signal[start:end] #Ventana a analizar
        segment = segment - np.mean(segment) #Resto la DC (asi elimino el pico inicial del espectro)

        # FFT
        N = window_size
        fft_result = 1/N*np.fft.fft(segment) #calculamos la fft normalizada
        freqs = fftfreq(len(segment), d=1/fs) #Vector con los bins de frecuencias
        power_spectrum = np.abs(fft_result)**2
        
        # Nos quedamos con la parte positiva del espectro
        pos_freqs = freqs[:len(freqs)//2]                       
        pos_power = power_spectrum[:len(power_spectrum)//2]

        mnf = mean_frequency(pos_power, pos_freqs)
        mdf = median_frequency(pos_power, pos_freqs)
        
        mnf_list.append(mnf)
        mdf_list.append(mdf)
        time_centers.append(time[start + window_size // 2]) #Tiempo medio de la ventana (frecuencia de cada bin)

    return np.array(time_centers), mnf_list, mdf_list

def stft_mnf_mdf(signal, time, fs, window_sec=4, step_sec=2, window_type='sinc'):

    # Conversión a muestras
    nperseg = int(window_sec * fs)
    noverlap = nperseg - int(step_sec * fs)

    # Calcular STFT
    f, t_stft, Zxx = stft(signal, fs=fs, window=window_type,
                          nperseg=nperseg, noverlap=noverlap)

    power = np.abs(Zxx) ** 2

    mnf_list = []
    mdf_list = []

    for i in range(power.shape[1]):  # columnas = ventanas temporales
        espectro = power[:, i]

        if np.sum(espectro) == 0:
            mnf_list.append(0)
            mdf_list.append(0)
            continue

        mnf = mean_frequency(espectro, f)
        mdf = median_frequency(espectro, f)

        mnf_list.append(mnf)
        mdf_list.append(mdf)

    # Asegurar correspondencia con el vector tiempo original (usamos centro de ventana real)
    # Como t_stft viene en segundos, no hace falta convertir
    return t_stft, mnf_list, mdf_list    

# //// Ver bien esta funcion ////
def calculate_linear_regression(times, values):

    slope, intercept, r_value, p_value, std_err = sp.stats.linregress(times, values)
    trend = slope * times + intercept

    return slope, intercept, r_value, p_value, std_err, trend

def windowed_rms(signal, time, fs, window_sec=4, step_sec=2):
    
    window_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)

    rms_list = []
    t_rms = []

    for start in range(0, len(signal) - window_samples + 1, step_samples):
        end = start + window_samples
        window = signal[start:end]
        rms = np.sqrt(np.mean(window ** 2))
        rms_list.append(rms)
        # Tiempo central de la ventana
        center_time = time[start:end].mean()
        t_rms.append(center_time)

    return np.array(t_rms), np.array(rms_list)  
 
# %% CARGA DE ARCHIVOS

# //// Carga de la matriz COMPLETA ////
# Rutas
dir_emg = r"C:\Users\santi\Desktop\Facu\APS\TP Final\EMG-Self-Perceived\sEMG_data"
dir_gt =  r"C:\Users\santi\Desktop\Facu\APS\TP Final\EMG-Self-Perceived\self_perceived_fatigue_index"

# Parámetros
subject_numbers = range(1, 7)  # Se define la ctd. de sujetos
EMG_index = [1, 3, 5, 7] #Columnas de los canales y los tiempos correspondientes del .csv
time_index = [0, 2, 4, 6]

# Diccionario
data_dict = {}

for subject_number in subject_numbers: #Recorremos los sujetos
    subject_key = f"subject_{subject_number}"
    data_dict[subject_key] = {} #Inicializa el diccionario en subject_X

    emg_folder = os.path.join(dir_emg, f"subject_{subject_number}")
    gt_folder  = os.path.join(dir_gt,  f"subject_{subject_number}")

    #Ordenamos los ensayos para cada sujeto
    emg_files = sorted([f for f in os.listdir(emg_folder) if f.endswith('.csv')], key=lambda x: int(re.findall(r'\d+', x)[-1]))
    
    for emg_file in emg_files: #Por cada .csv de emg_files (c/u de los ensayos)
        trial_number = int(re.findall(r'\d+', emg_file)[-1]) #Extraemos el numero de cada trial
        trial_key = f"trial_{trial_number}" 

        # Rutas completas hacia cada uno de los trials para el subject_X
        emg_path = os.path.join(emg_folder, emg_file)
        gt_path = os.path.join(gt_folder, emg_file)

        # Leemos EMG
        dataframe_emg = pd.read_csv(emg_path) #lee valores del .csv y los convierte en un data frame (parecido a un struct)
        ## Tenemos filas --> ctd. muestras
        ## Columnas --> variables medidas (Tiempo, Valores EMG)
        
        emg_data = dataframe_emg.iloc[:, EMG_index].values #Guardamos columnas de EMG (indices definidos al ppo.)
        time_data = dataframe_emg.iloc[:, time_index].values #Guardamos columnas de tiempo 

        # Leer etiquetas de indices de fatiga
        if os.path.exists(gt_path):
            dataframe_fatigue = pd.read_csv(gt_path) #Creamos data frame con los indices de fatiga
            label_times = dataframe_fatigue.iloc[:, 0].values #Primer columna del DF --> tiempos
            label_values = dataframe_fatigue.iloc[:, 1].values #Segunda columna del DF --> indices
        else:
            label_times = np.array([]) #De no existir los valores de fatiga, array vacio
            label_values = np.array([])

        # Creamos el diccionario
        data_dict[subject_key][trial_key] = {
            'emg': emg_data,
            'time': time_data,
            'labels': label_values,
            'label_times': label_times
        }
        
# //// Carga de la matriz unicamente musculos Agonistas ////
# Rutas
dir_emg = r"C:\Users\santi\Desktop\Facu\APS\TP Final\EMG-Self-Perceived\sEMG_data"
dir_gt =  r"C:\Users\santi\Desktop\Facu\APS\TP Final\EMG-Self-Perceived\self_perceived_fatigue_index"

# Parámetros
subject_numbers = range(1, 13)  # Se define la ctd. de sujetos
EMG_index = [1, 3, 5, 7]  # Columnas de los 4 canales EMG
time_index = [0, 2, 4, 6]  # Columnas de tiempo correspondientes
sample_freq = 1259

# Diccionario de salida
data_dict2 = {}

# Recorrer sujetos
for subject_number in subject_numbers:
    subject_key = f"subject_{subject_number}"
    data_dict2[subject_key] = {}

    emg_folder = os.path.join(dir_emg, f"subject_{subject_number}")
    gt_folder  = os.path.join(dir_gt,  f"subject_{subject_number}")

    emg_files = sorted(
        [f for f in os.listdir(emg_folder) if f.endswith('.csv')],
        key=lambda x: int(re.findall(r'\d+', x)[-1])
    )

    for emg_file in emg_files:
        trial_number = int(re.findall(r'\d+', emg_file)[-1])
        trial_key = f"trial_{trial_number}"
        emg_path = os.path.join(emg_folder, emg_file)
        gt_path = os.path.join(gt_folder, emg_file)

        # Leer .csv
        df_emg = pd.read_csv(emg_path)

        # Obtener índice del canal principal (agonista)
        index = get_agonista(trial_number)
        if index is None:
            continue  # saltar si no se encuentra un canal válido

        # Me quedo unicamente con las columnas del record ppal. 
        emg_column = EMG_index[index]
        time_column = time_index[index]

        emg_data = df_emg.iloc[:, emg_column].values
        time_data = df_emg.iloc[:, time_column].values

        # Leer etiquetas si existen
        if os.path.exists(gt_path):
            df_fatigue = pd.read_csv(gt_path)
            label_times = df_fatigue.iloc[:, 0].values
            label_values = df_fatigue.iloc[:, 1].values
        else:
            label_times = np.array([])
            label_values = np.array([])

        # Guardar en el diccionario
        data_dict2[subject_key][trial_key] = {
            'emg': emg_data,
            'time': time_data,
            'labels': label_values,
            'label_times': label_times
        }        
# %% PROCESAMIENTO

fs = 1259

# //// Filtrado ////
fc1, fc2 = 20, 450
b, a = signal.butter(4, [fc1, fc2], btype='bandpass', fs=fs)

# Copio el sujeto 1 para comparar
import copy
subject1_raw = {'subject_1': copy.deepcopy(data_dict['subject_1'])}
#plot_emg(subject1_raw,1,1)

#Dataset 1
for subject_key in data_dict:  # Recorremos los sujetos
    for trial_key in data_dict[subject_key]:  # Recorremos los ensayos
        emg_raw = data_dict[subject_key][trial_key]['emg']  # Matriz Nx4

        # Filtrado bidireccional canal por canal
        emg_filtered = np.zeros_like(emg_raw)
        for i in range(4):
            emg_filtered[:, i] = signal.filtfilt(b, a, emg_raw[:, i])

        # Reemplazo en el diccionario original
        data_dict[subject_key][trial_key]['emg'] = emg_filtered

# Dataset 2
for subject_key in data_dict2: #Recorremos los sujetos
    for trial_key in data_dict2[subject_key]:
        emg_raw = data_dict2[subject_key][trial_key]['emg']
        
        # Filtrado bidireccional para anular los problemas de fase
        emg_filtered = signal.filtfilt(b, a, emg_raw)
        
        # Reemplazo en el diccionario
        data_dict2[subject_key][trial_key]['emg'] = emg_filtered

subject = 'subject_1'
trial = 'trial_1'

# //// Ploteo comparativo ////
# Extraer señales
emg_raw = subject1_raw[subject][trial]['emg']       # Sin filtrar
emg_filt = data_dict[subject][trial]['emg']         # Filtrado
time = data_dict[subject][trial]['time']            # Tiempo (es el mismo para ambos)

# Gráfico comparativo por canal
plt.figure(figsize=(12, 10))
for i in range(4):
    plt.subplot(4, 1, i + 1)
    plt.plot(time[:, i], emg_raw[:, i], label='Raw', alpha=0.6)
    plt.plot(time[:, i], emg_filt[:, i], label='Filtrado')
    plt.title("Sujeto 1, Canal 1, Actividad 1")
    plt.ylabel("EMG (V)")
    plt.grid(True)
    if i == 0:
        plt.legend()

plt.tight_layout()
plt.xlabel("Tiempo (s)")
plt.show()

i = 3  # Canal 3 
# Gráfico comparativo para canal 3
plt.figure(figsize=(12, 5))
plt.plot(time[:, i], emg_raw[:, i], label='Raw', alpha=0.6)
plt.plot(time[:, i], emg_filt[:, i], label='Filtrado')
plt.title("Sujeto 1, Canal 3, Actividad 1")  # Título visible como Canal 3
plt.ylabel("EMG (V)")
plt.xlabel("Tiempo (s)")
plt.xlim([93.6, 94])  # Zoom en tiempo (ajustá si querés otro rango)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# //// Normalización ////

#Dataset 1
for subject_key in data_dict:
    for trial_key in data_dict[subject_key]:
        emg = data_dict[subject_key][trial_key]['emg']  # Señal ya filtrada, matriz Nx4
        
        max_vals = np.max(np.abs(emg), axis=0)

        emg_normalized = emg / max_vals  # Escalado por canal
        data_dict[subject_key][trial_key]['emg'] = emg_normalized
            
#Dataset 2
for subject_key in data_dict2:
    for trial_key in data_dict2[subject_key]:
        emg = data_dict2[subject_key][trial_key]['emg']  # Señal ya filtrada, matriz Nx4
        
        max_vals = np.max(np.abs(emg), axis=0)

        emg_normalized = emg / max_vals  # Escalado por canal
        data_dict2[subject_key][trial_key]['emg'] = emg_normalized

plot_emg_all(data_dict2, 1)


# %% ANALISIS FRECUENCIAL

# # //// MDF y MNF con sliding window ////
# fs = 1259
# trial = 'trial_1'

# mnf_all = []
# mdf_all = []

# plt.figure(figsize=(12, 5))
# for subject in range(1, 13):
#     subject_key = f"subject_{subject}"
#     emg = data_dict2[subject_key][trial]['emg'].flatten()  # Señal 1D
#     time = data_dict2[subject_key][trial]['time'].flatten()

#     t_mids, mnf, mdf = slidingwin_fft_mnf_mdf(emg, time, fs)

#     plt.plot(t_mids, mnf, label=f'{subject_key}')
#     # Guardar si querés usarlas después
#     mnf_all.append(mnf)
#     mdf_all.append(mdf)

# plt.xlabel("Tiempo (s)")
# plt.ylabel("Frecuencia (Hz)")
# plt.title("MNF por Sliding Window - Trial 1")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# //// MDF y MNF con STFT ////
fs = 1259
trial = 'trial_1'

plt.figure(figsize=(12, 5))

#MNF
for subject in range(1, 13):
    subject_key = f"subject_{subject}"
    emg = data_dict2[subject_key][trial]['emg'].flatten()
    time = data_dict2[subject_key][trial]['time'].flatten()

    t_stft, mnf, _ = stft_mnf_mdf(emg, time, fs, window_sec=4, step_sec=2)

    plt.plot(t_stft, mnf, label=f'{subject_key}')

plt.xlabel("Tiempo (s)")
plt.ylabel("Frecuencia (Hz)")
plt.title("MNF por STFT - Trial 1")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))

#MDF
for subject in range(1, 13):
    subject_key = f"subject_{subject}"
    emg = data_dict2[subject_key][trial]['emg'].flatten()
    time = data_dict2[subject_key][trial]['time'].flatten()

    t_stft, _, mdf = stft_mnf_mdf(emg, time, fs, window_sec=4, step_sec=2)

    plt.plot(t_stft, mdf, label=f'{subject_key}')

plt.xlabel("Tiempo (s)")
plt.ylabel("Frecuencia (Hz)")
plt.title("MDF por STFT - Trial 1")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# //// Subplots para los 6 trials ////

# ---- MNF ---- 

fs = 1259
window_sec = 4
step_sec = 2

# Nombres de los movimientos para los primeros 6 trials
movement_titles = [
    "Right Shoulder Flexion",
    "Left Shoulder Flexion",
    "Right Shoulder Extension",
    "Left Shoulder Extension",
    "Right Elbow Flexion",
    "Left Elbow Flexion"
]

fig, axs = plt.subplots(3, 2, figsize=(16, 10))
fig.suptitle("Mean Frequency (STFT) - Trials 1 to 6", fontsize=18)

for idx in range(6):
    row = idx // 2
    col = idx % 2
    ax = axs[row, col]

    trial = f"trial_{idx+1}"
    
    import string
    letter = string.ascii_lowercase[idx]  # (a), (b), ...

    for subject in range(1, 13):
        subject_key = f"subject_{subject}"
        emg = data_dict2[subject_key][trial]['emg'].flatten()
        time = data_dict2[subject_key][trial]['time'].flatten()

        t_stft, mnf, _ = stft_mnf_mdf(emg, time, fs, window_sec, step_sec)
        ax.plot(t_stft, mnf, label=f"{subject_key}", linewidth=1)

    ax.set_title(f"T{idx+1} - {movement_titles[idx]}", fontsize=11)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Mean Frequency (Hz)")
    ax.grid(True)

    # Letra del gráfico (a), (b), ...
    ax.text(0.01, 0.92, f"({letter})", transform=ax.transAxes, fontsize=12, fontweight='bold')

    # Leyenda en el borde derecho del subplot
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# ---- MDF ---- 

fig, axs = plt.subplots(3, 2, figsize=(16, 10))
fig.suptitle("Median Frequency (STFT) - Trials 1 to 6", fontsize=18)

for idx in range(6):
    row = idx // 2
    col = idx % 2
    ax = axs[row, col]

    trial = f"trial_{idx+1}"
    
    letter = string.ascii_lowercase[idx]  # (a), (b), ...

    for subject in range(1, 13):
        subject_key = f"subject_{subject}"
        emg = data_dict2[subject_key][trial]['emg'].flatten()
        time = data_dict2[subject_key][trial]['time'].flatten()

        t_stft, _, mdf = stft_mnf_mdf(emg, time, fs, window_sec, step_sec)
        ax.plot(t_stft, mdf, label=f"{subject_key}", linewidth=1)

    ax.set_title(f"T{idx+1} - {movement_titles[idx]}", fontsize=11)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Median Frequency (Hz)")
    ax.grid(True)

    # Letra del gráfico (a), (b), ...
    ax.text(0.01, 0.92, f"({letter})", transform=ax.transAxes, fontsize=12, fontweight='bold')

    # Leyenda en el borde derecho del subplot
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# ---- RMS ----

fig, axs = plt.subplots(3, 2, figsize=(16, 10))
fig.suptitle("Valores RMS (Sliding Window) - Trials 1 to 6", fontsize=18)

for idx in range(6):
    row = idx // 2
    col = idx % 2
    ax = axs[row, col]

    trial = f"trial_{idx+1}"
    
    letter = string.ascii_lowercase[idx]  # (a), (b), ...

    for subject in range(1, 13):
        subject_key = f"subject_{subject}"
        emg = data_dict2[subject_key][trial]['emg'].flatten()
        time = data_dict2[subject_key][trial]['time'].flatten()

        t_rms, rms = windowed_rms(emg, time, fs, window_sec, step_sec)
        ax.plot(t_rms, rms, label=f"{subject_key}", linewidth=1)

    ax.set_title(f"T{idx+1} - {movement_titles[idx]}", fontsize=11)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("valores RMS (µV)")
    ax.grid(True)

    # Letra del gráfico (a), (b), ...
    ax.text(0.01, 0.92, f"({letter})", transform=ax.transAxes, fontsize=12, fontweight='bold')

    # Leyenda en el borde derecho del subplot
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# %% TENDENCIAS LINEALES

# ---- MNF ----

fig, axs = plt.subplots(3, 2, figsize=(16, 10))
fig.suptitle("Regresión lineal de Mean Frequency (STFT) - Trials 1 to 6", fontsize=18)

mnf_slopes = {}  # Diccionario para guardar las pendientes

for idx in range(6):
    row = idx // 2
    col = idx % 2
    ax = axs[row, col]

    trial = f"trial_{idx+1}"
    letter = string.ascii_lowercase[idx]

    for subject in range(1, 13):
        subject_key = f"subject_{subject}"
        emg = data_dict2[subject_key][trial]['emg'].flatten()
        time = data_dict2[subject_key][trial]['time'].flatten()

        t_stft, mnf, _ = stft_mnf_mdf(emg, time, fs, window_sec, step_sec)

        # Calcular regresión
        slope, intercept, _, _, _, trend = calculate_linear_regression(t_stft, mnf)

        #Guardamos pendiente
        if subject_key not in mnf_slopes:
                mnf_slopes[subject_key] = {} #Creamos "subject key" en el diccionario
        mnf_slopes[subject_key][trial] = slope
        
        ax.plot(t_stft, trend, label=f"{subject_key} (slope={slope:.2f})", linewidth=1)

    ax.set_title(f"T{idx+1} - {movement_titles[idx]}", fontsize=11)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Mean Frequency (Hz)")
    ax.grid(True)
    ax.text(0.01, 0.92, f"({letter})", transform=ax.transAxes, fontsize=12, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=7)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# ---- MDF ----

fig, axs = plt.subplots(3, 2, figsize=(16, 10))
fig.suptitle("Regresión lineal de Median Frequency (STFT) - Trials 1 to 6", fontsize=18)

mdf_slopes = {}  # Diccionario para guardar las pendientes

for idx in range(6):
    row = idx // 2
    col = idx % 2
    ax = axs[row, col]

    trial = f"trial_{idx+1}"
    letter = string.ascii_lowercase[idx]

    for subject in range(1, 13):
        subject_key = f"subject_{subject}"
        emg = data_dict2[subject_key][trial]['emg'].flatten()
        time = data_dict2[subject_key][trial]['time'].flatten()

        t_stft, _, mdf = stft_mnf_mdf(emg, time, fs, window_sec, step_sec)

        # Calcular regresión
        slope, intercept, _, _, _, trend = calculate_linear_regression(t_stft, mdf)

        #Guardamos pendiente
        if subject_key not in mdf_slopes:
                mdf_slopes[subject_key] = {} #Creamos "subject key" en el diccionario
        mdf_slopes[subject_key][trial] = slope
        
        ax.plot(t_stft, trend, label=f"{subject_key} (slope={slope:.2f})", linewidth=1)

    ax.set_title(f"T{idx+1} - {movement_titles[idx]}", fontsize=11)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Median Frequency (Hz)")
    ax.grid(True)
    ax.text(0.01, 0.92, f"({letter})", transform=ax.transAxes, fontsize=12, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=7)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# ---- RMS ----

fig, axs = plt.subplots(3, 2, figsize=(16, 10))
fig.suptitle("Regresión lineal de RMS (Sliding Window) - Trials 1 to 6", fontsize=18)

rms_slopes = {}  # Diccionario para guardar las pendientes

for idx in range(6):
    row = idx // 2
    col = idx % 2
    ax = axs[row, col]

    trial = f"trial_{idx+1}"
    letter = string.ascii_lowercase[idx]

    for subject in range(1, 13):
        subject_key = f"subject_{subject}"
        emg = data_dict2[subject_key][trial]['emg'].flatten()
        time = data_dict2[subject_key][trial]['time'].flatten()

        t_rms, rms = windowed_rms(emg, time, fs, window_sec, step_sec)

        # Calcular regresión
        slope, intercept, _, _, _, trend = calculate_linear_regression(t_rms, rms)

        #Guardamos pendiente
        if subject_key not in rms_slopes:
                rms_slopes[subject_key] = {} #Creamos "subject key" en el diccionario
        rms_slopes[subject_key][trial] = slope
        
        ax.plot(t_rms, trend, label=f"{subject_key} (slope={slope:.2f})", linewidth=1)

    ax.set_title(f"T{idx+1} - {movement_titles[idx]}", fontsize=11)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("RMS (µV)")
    ax.grid(True)
    ax.text(0.01, 0.92, f"({letter})", transform=ax.transAxes, fontsize=12, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=7)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# %% RESULTADOS TENDENCIAS

# ---- MNF ----

fig, axs = plt.subplots(3, 2, figsize=(16, 10))
fig.suptitle("Regresión lineal de Mean Frequency (STFT) - Trials 1 to 6", fontsize=18)    

# Crear diccionario para guardar estadísticas
mnf_stats = {
    "trial": [],
    "mean_slope": [],
    "std_slope": [],
    "sem_slope": [],  # SEM = std / sqrt(n)
}

# Recorremos cada trial 
for i in range(1, 7):
    trial_key = f"trial_{i}"
    slopes = []

    for subject_key in mnf_slopes:
        slope_val = mnf_slopes[subject_key][trial_key]
        slopes.append(slope_val)

    slopes = np.array(slopes)

    mnf_stats["trial"].append(trial_key)
    mnf_stats["mean_slope"].append(np.mean(slopes)) #Promedio
    mnf_stats["std_slope"].append(np.std(slopes, ddof=1)) #Desvio estandar
    mnf_stats["sem_slope"].append(np.std(slopes, ddof=1) / np.sqrt(len(slopes))) #Incertidumbre (error estandar de la media)

# Convertir a DataFrame
mnf_stats_df = pd.DataFrame(mnf_stats)

# Mostrar
print(mnf_stats_df)

plt.figure(figsize=(10, 5))
plt.bar(mnf_stats_df["trial"], mnf_stats_df["mean_slope"], 
        yerr=mnf_stats_df["sem_slope"], capsize=5, color='skyblue')
plt.axhline(0, color='gray', linestyle='--')
plt.title("Pendiente media de MNF por ensayo ± SEM")
plt.ylabel("Pendiente (Hz/s)")
plt.xlabel("Ensayo")
plt.grid(True, axis='y')
plt.show()

# ---- MDF ----

# Crear diccionario para guardar estadísticas
mdf_stats = {
    "trial": [],
    "mean_slope": [],
    "std_slope": [],
    "sem_slope": [],  # SEM = std / sqrt(n)[]
}

# Recorremos cada trial 
for i in range(1, 7):
    trial_key = f"trial_{i}"
    slopes = []

    for subject_key in mdf_slopes:
        slope_val = mdf_slopes[subject_key][trial_key]
        slopes.append(slope_val)

    slopes = np.array(slopes)

    mdf_stats["trial"].append(trial_key)
    mdf_stats["mean_slope"].append(np.mean(slopes)) #Promedio
    mdf_stats["std_slope"].append(np.std(slopes, ddof=1)) #Desvio estandar
    mdf_stats["sem_slope"].append(np.std(slopes, ddof=1) / np.sqrt(len(slopes))) #Incertidumbre (error estandar de la media)

# Convertir a DataFrame
mdf_stats_df = pd.DataFrame(mdf_stats)

# Mostrar
print(mdf_stats_df)

plt.figure(figsize=(10, 5))
plt.bar(mdf_stats_df["trial"], mdf_stats_df["mean_slope"], 
        yerr=mdf_stats_df["sem_slope"], capsize=5, color='skyblue')
plt.axhline(0, color='gray', linestyle='--')
plt.title("Pendiente media de MDF por ensayo ± SEM")
plt.ylabel("Pendiente (Hz/s)")
plt.xlabel("Ensayo")
plt.grid(True, axis='y')
plt.show()

# ---- RMS ----

# Crear diccionario para guardar estadísticas
rms_stats = {
    "trial": [],
    "mean_slope": [],
    "std_slope": [],
    "sem_slope": [],  # SEM = std / sqrt(n)[]
}

# Recorremos cada trial 
for i in range(1, 7):
    trial_key = f"trial_{i}"
    slopes = []

    for subject_key in rms_slopes:
        slope_val = rms_slopes[subject_key][trial_key]
        slopes.append(slope_val)

    slopes = np.array(slopes)

    rms_stats["trial"].append(trial_key)
    rms_stats["mean_slope"].append(np.mean(slopes)) #Promedio
    rms_stats["std_slope"].append(np.std(slopes, ddof=1)) #Desvio estandar
    rms_stats["sem_slope"].append(np.std(slopes, ddof=1) / np.sqrt(len(slopes))) #Incertidumbre (error estandar de la media)

# Convertir a DataFrame
rms_stats_df = pd.DataFrame(rms_stats)

# Mostrar
print(rms_stats_df)

plt.figure(figsize=(10, 5))
plt.bar(rms_stats_df["trial"], rms_stats_df["mean_slope"], 
        yerr=rms_stats_df["sem_slope"], capsize=5, color='skyblue')
plt.axhline(0, color='gray', linestyle='--')
plt.title("Pendiente media de RMS por ensayo ± SEM")
plt.ylabel("Pendiente (Hz/s)")
plt.xlabel("Ensayo")
plt.grid(True, axis='y')
plt.show()

# %% CORRELACION CON EL SPF, PEARSON

# ---- Regresion SPF ----

fig, axs = plt.subplots(3, 2, figsize=(16, 10))
fig.suptitle("Regresión lineal de RMS (Sliding Window) - Trials 1 to 6", fontsize=18)

spf_slopes = {}

for idx in range(6):
    row = idx // 2
    col = idx % 2
    ax = axs[row, col]

    trial = f"trial_{idx+1}"
    letter = string.ascii_lowercase[idx]

    for subject in range(1, 13):
        subject_key = f"subject_{subject}"
        labels = data_dict2[subject_key][trial]['labels'].flatten()
        labels_time = data_dict2[subject_key][trial]['label_times'].flatten()
        
        # Buscar primeros tiempos donde aparece 1 y 2
        t0 = labels_time[0] #principio
        t1 = labels_time[np.where(labels == 1)[0][0]]
        t2 = labels_time[np.where(labels == 2)[0][0]]
        t3 = labels_time[-1] #final

        t_spf = np.array([t0, t1, t2, t3])
        spf = np.array([0, 1, 2, 3]) #De alguna manera cree el nivel 3, donde el paciente llega al fallo del ejercicio

        # Regressión lineal simple entre niveles de fatiga y tiempo
        slope, intercept, _, _, _, trend = calculate_linear_regression(t_spf, spf)

        # Guardar pendiente
        if subject_key not in spf_slopes:
            spf_slopes[subject_key] = {}
        spf_slopes[subject_key][trial] = slope

        ax.plot(t_spf, trend, label=f"{subject_key} (slope={slope:.2f})", linewidth=1)

    ax.set_title(f"T{idx+1} - {movement_titles[idx]}", fontsize=11)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("SPF")
    ax.grid(True)
    ax.text(0.01, 0.92, f"({letter})", transform=ax.transAxes, fontsize=12, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=7)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# ---- Pearson ----
from scipy.stats import pearsonr

# Diccionarios para guardar resultados
mnf_stats = {"trial": [], "pearson_r": [], "pearson_p": []}
mdf_stats = {"trial": [], "pearson_r": [], "pearson_p": []}
rms_stats = {"trial": [], "pearson_r": [], "pearson_p": []}

# Función para calcular correlación por trial
def correlate_by_trial(spf_slopes, emg_slopes, stats_dict):
    for trial_idx in range(6):
        trial_key = f"trial_{trial_idx+1}"

        spf_vals = []
        emg_vals = []

        for subject in range(1, 13):
            subject_key = f"subject_{subject}"
            try:
                spf_vals.append(spf_slopes[subject_key][trial_key])
                emg_vals.append(emg_slopes[subject_key][trial_key])
            except KeyError:
                continue  # Por si falta algún dato

        if len(spf_vals) >= 2:
            r, p = pearsonr(spf_vals, emg_vals)
        else:
            r, p = np.nan, np.nan

        stats_dict["trial"].append(trial_key)
        stats_dict["pearson_r"].append(r)
        stats_dict["pearson_p"].append(p)

# Ejecutar para cada métrica
correlate_by_trial(spf_slopes, mnf_slopes, mnf_stats)
correlate_by_trial(spf_slopes, mdf_slopes, mdf_stats)
correlate_by_trial(spf_slopes, rms_slopes, rms_stats)

# Convertir cada diccionario en DataFrame
df_mnf = pd.DataFrame(mnf_stats)
df_mdf = pd.DataFrame(mdf_stats)
df_rms = pd.DataFrame(rms_stats)

# Renombrar columnas para identificar la métrica
df_mnf.columns = ['Trial', 'MNF_r', 'MNF_p']
df_mdf.columns = ['Trial', 'MDF_r', 'MDF_p']
df_rms.columns = ['Trial', 'RMS_r', 'RMS_p']

# Unir las tres tablas en una sola
results_table = pd.concat([df_mnf, df_mdf.iloc[:, 1:], df_rms.iloc[:, 1:]], axis=1)

# Mostrar la tabla
print(results_table.to_string(index=False))
