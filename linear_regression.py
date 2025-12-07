import numpy as np
# Visualizzazione grafica
from visualization import plot_training_history

def load_data_from_csv(filename):
    """Carica i dati dal CSV in un array numpy"""
    data_list = []

    try: 
      with open(filename, 'r') as file:
        next(file) # Salta l'intestazione
        for line in file:
          ore_studio, voto_finale = line.strip().split(',')
          data_list.append([float(ore_studio), float(voto_finale)])
    except IOError:
      print(f"Errore nel caricamento del file {filename}")
      return None
    data_array = np.array(data_list)
    X = data_array[:, 0]
    Y = data_array[:, 1]
    return X, Y
    

X, y = load_data_from_csv('dati_voti.csv')
m = len(X)
w = 0.0
b = 0.0
learning_rate = 0.01
num_iterations = 5000

print(f"Numero di dati: {m}")
print(f"Primi 5 dati: {X[:5]}")
print(f"Primi 5 dati: {y[:5]}")


def compute_cost(X, y, w, b, m):
  """Calcola il costo J(w, b)"""
  cost = 0
  for i in range(m):
    f_x = (w * X[i]) + b
    cost += (f_x - y[i]) ** 2  # Accumula invece di sovrascrivere
  total_cost = (1.0 / (2 * m)) * cost  # Corretto: 1/(2*m) non (1/2)*m
  return total_cost

def compute_gradient(X, y, w, b, m):
  dj_dw = 0
  dj_db = 0

  for i in range(m):
    f_x = w * X[i] + b
    dj_dw += (f_x - y[i]) * X[i]  # Corretto: moltiplicazione e accumulo
    dj_db += f_x - y[i]  # Corretto: accumulo
  dj_dw = dj_dw / m
  dj_db = dj_db / m
  return dj_dw, dj_db

def gradient_descent(X, y, w, b, learning_rate, num_iterations, m):
  tmp_w = w
  tmp_b = b
  history_j = []
  history_params = []

  for i in range(num_iterations):
    dj_dw, dj_db = compute_gradient(X, y, tmp_w, tmp_b, m)  # Usa tmp_w, tmp_b non w, b
    tmp_w = tmp_w - learning_rate * dj_dw
    tmp_b = tmp_b - learning_rate * dj_db
    if i < num_iterations:
      history_j.append(compute_cost(X, y, tmp_w, tmp_b, m))
      history_params.append([tmp_w, tmp_b])
  
  return tmp_w, tmp_b, history_j, history_params

def predict(X, w, b):
  y_hat = w*X + b
  return y_hat

final_w, final_b, history_j, history_params = gradient_descent(X, y, w, b, learning_rate, num_iterations, m)

print (f"final w and b = {final_w}, {final_b}")
total_cost = compute_cost(X, y, final_w, final_b, m)
print(f"final total cost: {total_cost}")

input_x = 8.52
result = predict(input_x, final_w, final_b)

print(f"the prediction for {input_x} hours of study is this vote: {result}")


plot_training_history(history_j, history_params, X, y, final_w, final_b)


