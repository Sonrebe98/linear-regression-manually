from copy import deepcopy
import numpy as np
import os
import matplotlib.pyplot as plt
from visualization import plot_features_vs_target, plot_training_history

def load_data_from_csv(filename):
    """Carica i dati dal CSV in un array numpy"""
    data_list = []

    try: 
      with open(filename, 'r') as file:
        next(file) # Salta l'intestazione
        for line in file:
          features = line.strip().split(',')
          data_list.append([float(feature) for feature in features])
    except (IOError, OSError, FileNotFoundError) as e:
      print(f"Errore nel caricamento del file {filename}: {e}")
      return None
    except Exception as e:
      print(f"Errore imprevisto durante il caricamento: {e}")
      return None
    
    if not data_list:
      print("Il file è vuoto o non contiene dati validi")
      return None
      
    data_array = np.array(data_list)
    X = data_array[:, :-1]  # Tutte le colonne tranne l'ultima (features)
    Y = data_array[:, -1]   # Ultima colonna (target)
    return X, Y

# il taining set è stato scaricato da kagglehub e si trova nella directory dello script: https://www.kaggle.com/datasets/prokshitha/home-value-insights?resource=download
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'house_price_regression_dataset.csv')

# Leggi i nomi delle features dal CSV
with open(csv_path, 'r') as file:
    header = file.readline().strip()
    feature_names = header.split(',')[:-1]  # Tutte le colonne tranne l'ultima (target)

X, y = load_data_from_csv(csv_path)


def feature_scaling(X):
  """Normalizza le features usando mean normalization"""
  average = X.mean(axis=0)
  range = X.max(axis=0) - X.min(axis=0)
  # Evita divisione per zero
  range[range == 0] = 1
  return (X - average) / range, average, range

def target_scaling(y):
  """Normalizza il target usando mean normalization"""
  average = y.mean()
  range = y.max() - y.min()
  if range == 0:
    range = 1
  return (y - average) / range, average, range

# Salva i dati originali per la denormalizzazione
X_original = X.copy()
y_original = y.copy()

# Normalizza X e y
X, X_mean, X_range = feature_scaling(X)
y, y_mean, y_range = target_scaling(y)

print(X.shape)
print(X[:5])
print(y.shape)

# plot_features_vs_target(X, y, feature_names=feature_names)
# plt.show()

def compute_cost(X, y, w, b, m):
  cost = 0 

  for i in range(m):
    f_xi = np.dot(X[i], w) + b
    cost += (f_xi - y[i])**2
  cost = (1 / (2*m)) * cost 

  return cost


b = 0
w = np.zeros(X.shape[1])

cost = compute_cost(X, y, w, b, len(X))
print(f"Costo iniziale (su dati normalizzati): {cost:.6f}")

def compute_gradient(X, y, w, b, m):
  dj_dw = np.zeros(w.shape)
  dj_db = 0

  for i in range(m):
    f_xi = np.dot(X[i], w) + b
    for j in range(len(w)):
      dj_dw[j] += (f_xi - y[i]) * X[i, j]
    dj_db += (f_xi - y[i])

  dj_dw = dj_dw / m
  dj_db = dj_db / m
  return dj_dw, dj_db

dj_dw, dj_db = compute_gradient(X, y, w, b, len(X))
print(f"Gradiente: {dj_dw}")
print(f"Gradiente: {dj_db}")

learning_rate = 0.1
num_iterations = 10000

def gradient_descent(X, y, w, b, learning_rate, num_iterations, m):
  tmp_w = deepcopy(w.copy())
  tmp_b = b
  history_j = []
  history_params = []

  for i in range(num_iterations):
    dj_dw, dj_db = compute_gradient(X, y, tmp_w, tmp_b, m)
    tmp_w = tmp_w - learning_rate * dj_dw
    tmp_b = tmp_b - learning_rate * dj_db
    
    # Calcola e salva il costo ad ogni iterazione
    current_cost = compute_cost(X, y, tmp_w, tmp_b, m)
    history_j.append(current_cost)
    history_params.append([tmp_w.copy(), tmp_b])
    
    # Controlla convergenza (confronta con l'iterazione precedente)
    if i > 0 and abs(history_j[i] - history_j[i-1]) < 1e-6:
      print(f"Convergenza raggiunta dopo {i+1} iterazioni")
      print(f"Costo finale (normalizzato): {history_j[i]:.6f}")
      break
    
    # Stampa progresso ogni 1000 iterazioni
    if (i + 1) % 1000 == 0:
      print(f"Iterazione {i+1}/{num_iterations}, Costo: {current_cost:.6f}")
  
  if i == num_iterations - 1:
    print(f"Numero massimo di iterazioni raggiunto: {num_iterations}")
    print(f"Costo finale (normalizzato): {history_j[-1]:.6f}")
  
  return tmp_w, tmp_b, history_j, history_params

final_w, final_b, history_j, history_params = gradient_descent(X, y, w, b, learning_rate, num_iterations, len(X))

print(f"\nParametri finali (su dati normalizzati):")
print(f"w: {final_w}")
print(f"b: {final_b}")

# Funzione per fare predizioni e denormalizzarle
def predict(X_input, w, b, X_mean, X_range, y_mean, y_range):
    """Fa predizioni e le denormalizza per ottenere valori in scala originale"""
    # Normalizza l'input
    X_normalized = (X_input - X_mean) / X_range
    # Predizione su dati normalizzati
    y_pred_normalized = np.dot(X_normalized, w) + b
    # Denormalizza la predizione
    y_pred = y_pred_normalized * y_range + y_mean
    return y_pred

# Test con alcuni esempi
print(f"\nEsempi di predizioni:")
for i in range(min(5, len(X_original))):
    pred = predict(X_original[i:i+1], final_w, final_b, X_mean, X_range, y_mean, y_range)
    print(f"Esempio {i+1}: Predetto = {pred[0]:.2f}, Reale = {y_original[i]:.2f}, Errore = {abs(pred[0] - y_original[i]):.2f}")

plot_training_history(history_j, history_params, X, y, final_w, final_b,
                      X_original=X_original, y_original=y_original,
                      X_mean=X_mean, X_range=X_range,
                      y_mean=y_mean, y_range=y_range,
                      feature_names=feature_names)
plt.show()