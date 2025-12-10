import matplotlib.pyplot as plt
import numpy as np

def plot_features_vs_target(X, y, feature_names=None, figsize=(15, 10)):
    """
    Visualizza tutte le features rispetto al target in subplots.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Matrice delle features (n_samples, n_features)
    y : numpy.ndarray
        Array del target (n_samples,)
    feature_names : list, optional
        Lista dei nomi delle features. Se None, usa nomi generici.
    figsize : tuple, optional
        Dimensione della figura (width, height)
    """
    n_features = X.shape[1]
    
    # Calcola il numero di righe e colonne per i subplots
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols  # Arrotonda per eccesso
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]
    
    # Se non sono forniti i nomi delle features, usa nomi generici
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(n_features)]
    
    for i in range(n_features):
        ax = axes[i]
        ax.scatter(X[:, i], y, alpha=0.5, s=20)
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel('Target (House Price)')
        ax.set_title(f'{feature_names[i]} vs House Price')
        ax.grid(True, alpha=0.3)
    
    # Nasconde i subplots non utilizzati
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig

def plot_single_feature_vs_target(X, y, feature_idx, feature_name=None, figsize=(8, 6)):
    """
    Visualizza una singola feature rispetto al target.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Matrice delle features (n_samples, n_features)
    y : numpy.ndarray
        Array del target (n_samples,)
    feature_idx : int
        Indice della feature da visualizzare
    feature_name : str, optional
        Nome della feature. Se None, usa 'Feature {feature_idx}'
    figsize : tuple, optional
        Dimensione della figura (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if feature_name is None:
        feature_name = f'Feature {feature_idx}'
    
    ax.scatter(X[:, feature_idx], y, alpha=0.5, s=20)
    ax.set_xlabel(feature_name)
    ax.set_ylabel('Target (House Price)')
    ax.set_title(f'{feature_name} vs House Price')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_target_distribution(y, figsize=(8, 6)):
    """
    Visualizza la distribuzione del target.
    
    Parameters:
    -----------
    y : numpy.ndarray
        Array del target (n_samples,)
    figsize : tuple, optional
        Dimensione della figura (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.hist(y, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('House Price')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of House Prices')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

def plot_features_distribution(X, feature_names=None, figsize=(15, 10)):
    """
    Visualizza la distribuzione di tutte le features.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Matrice delle features (n_samples, n_features)
    feature_names : list, optional
        Lista dei nomi delle features. Se None, usa nomi generici.
    figsize : tuple, optional
        Dimensione della figura (width, height)
    """
    n_features = X.shape[1]
    
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]
    
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(n_features)]
    
    for i in range(n_features):
        ax = axes[i]
        ax.hist(X[:, i], bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {feature_names[i]}')
        ax.grid(True, alpha=0.3, axis='y')
    
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig

def plot_training_history(history_j, history_params, X, y, final_w, final_b, 
                          X_original=None, y_original=None, X_mean=None, X_range=None, 
                          y_mean=None, y_range=None, feature_names=None):
    """
    Visualizza la storia del training e i risultati finali
    
    Parameters:
    -----------
    history_j: lista dei costi durante il training
    history_params: lista di [w, b] durante il training
    X: dati di input normalizzati (features)
    y: dati di output normalizzati (target)
    final_w: peso finale
    final_b: intercetta finale
    X_original: dati originali non normalizzati (opzionale, per predizioni)
    y_original: target originali non normalizzati (opzionale, per confronto)
    X_mean: media delle features per denormalizzazione (opzionale)
    X_range: range delle features per denormalizzazione (opzionale)
    y_mean: media del target per denormalizzazione (opzionale)
    y_range: range del target per denormalizzazione (opzionale)
    feature_names: lista dei nomi delle features (opzionale)
    """
    n_features = X.shape[1]
    
    # Crea una figura con più subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Grafico del costo durante il training
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(history_j, 'b-', linewidth=2)
    ax1.set_xlabel('Iterazione', fontsize=12)
    ax1.set_ylabel('Costo J(w, b)', fontsize=12)
    ax1.set_title('Evoluzione del Costo durante il Training', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Scala logaritmica per vedere meglio la convergenza
    
    # 2. Grafico dell'evoluzione dei parametri w
    ax2 = plt.subplot(2, 3, 2)
    if feature_names is None:
        feature_names = [f'w{i+1}' for i in range(n_features)]
    
    # Estrai i valori di w per ogni iterazione
    w_history = np.array([params[0] for params in history_params])
    for i in range(min(n_features, 7)):  # Mostra al massimo 7 features per leggibilità
        ax2.plot(w_history[:, i], label=feature_names[i], linewidth=1.5, alpha=0.7)
    ax2.set_xlabel('Iterazione', fontsize=12)
    ax2.set_ylabel('Valore del Parametro w', fontsize=12)
    ax2.set_title('Evoluzione dei Parametri w', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=8, loc='best')
    ax2.grid(True, alpha=0.3)
    
    # 3. Grafico dell'evoluzione del parametro b
    ax3 = plt.subplot(2, 3, 3)
    b_history = np.array([params[1] for params in history_params])
    ax3.plot(b_history, 'r-', linewidth=2)
    ax3.set_xlabel('Iterazione', fontsize=12)
    ax3.set_ylabel('Valore del Parametro b', fontsize=12)
    ax3.set_title('Evoluzione del Parametro b', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Grafico Predizioni vs Valori Reali (se disponibili i dati originali)
    if X_original is not None and y_original is not None and all(p is not None for p in [X_mean, X_range, y_mean, y_range]):
        ax4 = plt.subplot(2, 3, 4)
        
        # Calcola predizioni per tutti i dati
        X_normalized = (X_original - X_mean) / X_range
        y_pred_normalized = np.dot(X_normalized, final_w) + final_b
        y_pred = y_pred_normalized * y_range + y_mean
        
        # Prendi un campione per visualizzazione (max 1000 punti)
        sample_size = min(1000, len(y_original))
        indices = np.random.choice(len(y_original), sample_size, replace=False)
        
        ax4.scatter(y_original[indices], y_pred[indices], alpha=0.5, s=20)
        # Linea ideale (predizione perfetta)
        min_val = min(y_original.min(), y_pred.min())
        max_val = max(y_original.max(), y_pred.max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predizione perfetta')
        ax4.set_xlabel('Valore Reale (House Price)', fontsize=12)
        ax4.set_ylabel('Valore Predetto (House Price)', fontsize=12)
        ax4.set_title('Predizioni vs Valori Reali', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4 = plt.subplot(2, 3, 4)
        ax4.text(0.5, 0.5, 'Dati originali non disponibili\nper la visualizzazione', 
                ha='center', va='center', fontsize=12, transform=ax4.transAxes)
        ax4.set_title('Predizioni vs Valori Reali', fontsize=14, fontweight='bold')
    
    # 5. Grafico degli Errori (Residui)
    if X_original is not None and y_original is not None and all(p is not None for p in [X_mean, X_range, y_mean, y_range]):
        ax5 = plt.subplot(2, 3, 5)
        
        errors = np.abs(y_original - y_pred)
        ax5.scatter(y_original[indices], errors[indices], alpha=0.5, s=20, color='red')
        ax5.set_xlabel('Valore Reale (House Price)', fontsize=12)
        ax5.set_ylabel('Errore Assoluto', fontsize=12)
        ax5.set_title('Distribuzione degli Errori', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # Aggiungi statistiche
        mean_error = errors.mean()
        ax5.axhline(y=mean_error, color='green', linestyle='--', linewidth=2, 
                   label=f'Errore medio: {mean_error:.2f}')
        ax5.legend()
    else:
        ax5 = plt.subplot(2, 3, 5)
        ax5.text(0.5, 0.5, 'Dati originali non disponibili\nper la visualizzazione', 
                ha='center', va='center', fontsize=12, transform=ax5.transAxes)
        ax5.set_title('Distribuzione degli Errori', fontsize=14, fontweight='bold')
    
    # 6. Istogramma degli errori
    if X_original is not None and y_original is not None and all(p is not None for p in [X_mean, X_range, y_mean, y_range]):
        ax6 = plt.subplot(2, 3, 6)
        ax6.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
        ax6.set_xlabel('Errore Assoluto', fontsize=12)
        ax6.set_ylabel('Frequenza', fontsize=12)
        ax6.set_title('Distribuzione degli Errori (Istogramma)', fontsize=14, fontweight='bold')
        ax6.axvline(x=mean_error, color='red', linestyle='--', linewidth=2, 
                   label=f'Errore medio: {mean_error:.2f}')
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
    else:
        ax6 = plt.subplot(2, 3, 6)
        ax6.text(0.5, 0.5, 'Dati originali non disponibili\nper la visualizzazione', 
                ha='center', va='center', fontsize=12, transform=ax6.transAxes)
        ax6.set_title('Distribuzione degli Errori (Istogramma)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_final_results(X, y, final_w, final_b):
    """
    Visualizza i risultati finali
    
    Parameters:
    -----------
    X: dati di input (features)
    y: dati di output (target)
    final_w: peso finale
    final_b: intercetta finale
    """
    fig = plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(X, y, 'b-', linewidth=2)
    plt.xlabel('Feature', fontsize=12)
    plt.ylabel('Target', fontsize=12)
    plt.title('Dati e Retta di Regressione Finale', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
