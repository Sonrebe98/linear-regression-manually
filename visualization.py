import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(history_j, history_params, X, y, final_w, final_b):
    """
    Visualizza la storia del training e i risultati finali
    
    Args:
        history_j: lista dei costi durante il training
        history_params: lista di [w, b] durante il training
        X: dati di input (ore di studio)
        y: dati di output (voti)
        final_w: peso finale
        final_b: intercetta finale
    """
    # Crea una figura con 3 subplot
    fig = plt.figure(figsize=(15, 5))
    
    # 1. Grafico del costo nel tempo
    plt.subplot(1, 3, 1)
    iterations = range(len(history_j))
    plt.plot(iterations, history_j, 'b-', linewidth=2)
    plt.xlabel('Iterazione', fontsize=12)
    plt.ylabel('Costo J(w, b)', fontsize=12)
    plt.title('Evoluzione del Costo durante il Training', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 2. Grafico dell'evoluzione dei parametri w e b
    plt.subplot(1, 3, 2)
    w_history = [params[0] for params in history_params]
    b_history = [params[1] for params in history_params]
    plt.plot(iterations, w_history, 'r-', linewidth=2, label='w (pendenza)')
    plt.plot(iterations, b_history, 'g-', linewidth=2, label='b (intercetta)')
    plt.xlabel('Iterazione', fontsize=12)
    plt.ylabel('Valore del Parametro', fontsize=12)
    plt.title('Evoluzione dei Parametri w e b', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 3. Grafico dei dati con la retta di regressione finale
    plt.subplot(1, 3, 3)
    # Plot dei dati
    plt.scatter(X, y, alpha=0.5, s=20, color='blue', label='Dati reali')
    
    # Plot della retta di regressione finale
    X_line = np.linspace(X.min(), X.max(), 100)
    y_line = final_w * X_line + final_b
    plt.plot(X_line, y_line, 'r-', linewidth=3, label=f'Regressione: y = {final_w:.2f}x + {final_b:.2f}')
    
    plt.xlabel('Ore di Studio', fontsize=12)
    plt.ylabel('Voto Finale', fontsize=12)
    plt.title('Dati e Retta di Regressione Finale', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Mostra tutti i grafici
    plt.tight_layout()
    plt.show()
    
    # Stampa statistiche
    print("\n" + "="*60)
    print("STATISTICHE DEL TRAINING")
    print("="*60)
    print(f"Costo iniziale: {history_j[0]:.2f}")
    print(f"Costo finale: {history_j[-1]:.2f}")
    print(f"Riduzione del costo: {((history_j[0] - history_j[-1]) / history_j[0] * 100):.2f}%")
    print(f"\nParametro w iniziale: {w_history[0]:.2f}")
    print(f"Parametro w finale: {w_history[-1]:.2f}")
    print(f"\nParametro b iniziale: {b_history[0]:.2f}")
    print(f"Parametro b finale: {b_history[-1]:.2f}")
    print("="*60)

