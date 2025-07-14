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

# def stft_mnf_mdf(signal, fs=1259, window='hann', nperseg= 1000, noverlap=256):
  
#     f_stft, t_stft, Zxx = stft(signal, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
#     power = np.abs(Zxx) ** 2  # Espectro de potencia, Zxx matriz frecuencias x tiempo

#     mnf_list = []
#     mdf_list = []

#     for i in range(power.shape[1]):  # columnas = ventanas temporales
#         espectro = power[:, i] #tomamos todas las frecuencias de la ventana temporal i

#         #Agregue esta bloque porque no le gustaba calcular la mediana cuando la suma era cercana a cero
#         if np.sum(espectro) == 0:
#             mnf_list.append(0)
#             mdf_list.append(0)
#             continue
        
#         # MNF y MDF
#         mnf = mean_frequency(espectro, f_stft)
#         mdf = median_frequency(espectro, f_stft)

#         mnf_list.append(mnf)
#         mdf_list.append(mdf)

#     return t_stft, mnf_list, mdf_list, f_stft, Zxx   
 
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
plt.title("Sujeto 1, Canal 4, Actividad 1")  # Título visible como Canal 4
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

# %% RTA. EN FRECUENCIA

#FFT
trial = 1
plot_fft_all_subjects(data_dict2, fs, trial)

# %% MAIN

# //// MDF y MNF con sliding window ////
fs = 1259
trial = 'trial_1'

mnf_all = []
mdf_all = []

plt.figure(figsize=(12, 5))
for subject in range(1, 13):
    subject_key = f"subject_{subject}"
    emg = data_dict2[subject_key][trial]['emg'].flatten()  # Señal 1D
    time = data_dict2[subject_key][trial]['time'].flatten()

    t_mids, mnf, mdf = slidingwin_fft_mnf_mdf(emg, time, fs)

    plt.plot(t_mids, mnf, label=f'{subject_key}')
    # Guardar si querés usarlas después
    mnf_all.append(mnf)
    mdf_all.append(mdf)

plt.xlabel("Tiempo (s)")
plt.ylabel("Frecuencia (Hz)")
plt.title("MNF por Sliding Window - Trial 1")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

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



# Espectrograma

subject_key = f"subject_{2}"

emg = data_dict2[subject_key][trial]['emg'].flatten()

t_stft, mnf_list, mdf_list, f_stft, Zxx = stft_mnf_mdf(emg)

plt.figure() 

pcm = plt.pcolormesh(t_stft, f_stft, np.abs(Zxx), shading='gouraud')
plt.title("STFT (Espectrograma)")
plt.ylabel("Frecuencia [Hz]")
plt.xlabel("Tiempo [s]")
    
# Colorbar en eje externo
cbar_ax = plt.axes([0.92, 0.11, 0.015, 0.35])  # [left, bottom, width, height]
plt.colorbar(pcm, cax=cbar_ax, label="Magnitud")
    
plt.tight_layout(rect=[0, 0, 0.9, 1])  # Dejar espacio para colorbar a la derecha
plt.show()

plot_fft_all_subjects(data_dict2, 1259, 1)

#%% estimo el espectro con welch

(fwelch_ecgn, pxx_ecgn) = sp.signal.welch(emg, fs ,nfft = len(emg), window = 'hamming', nperseg = len(emg)//6, axis = 0)

#Ecg con ruido
plt.figure()
bfrec = fwelch_ecgn <= fs/2
plt.plot(fwelch_ecgn, 10*np.log10(2*np.abs(pxx_ecgn[bfrec])**2))
plt.ylabel('Densidad de Potencia [dB]')
plt.xlabel('Frecuencia [Hz]')
plt.title("Representación espectral - ECG Con Ruido")
