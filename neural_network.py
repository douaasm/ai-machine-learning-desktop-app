import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def lancer_reseau_neurones():
    try:
        min_x1 = float(entry_x1_min.get())
        max_x1 = float(entry_x1_max.get())
        min_x2 = float(entry_x2_min.get())
        max_x2 = float(entry_x2_max.get())
        n = int(entry_n.get())
        hidden = int(entry_hidden.get())
        iterations = int(entry_iterations.get())

        if min_x1 >= max_x1 or min_x2 >= max_x2:
            raise ValueError("Les valeurs minimales doivent être inférieures aux valeurs maximales.")
        if n < 20:
            raise ValueError("La taille de l'échantillon doit être au moins égale à 20.")
        if hidden < 1 or iterations < 1:
            raise ValueError("Les paramètres du modèle doivent être strictement positifs.")

        np.random.seed(42)
        X1 = np.random.uniform(min_x1, max_x1, n)
        X2 = np.random.uniform(min_x2, max_x2, n)
        y = 0.7 * X1**2 - 1.2 * X2 + 3 * np.sin(X1) + np.random.normal(0, 2, n)
        X = np.column_stack((X1, X2))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        model = MLPRegressor(
            hidden_layer_sizes=(hidden,),
            activation="relu",
            solver="adam",
            max_iter=iterations,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        for widget in frame_graph.winfo_children():
            widget.destroy()

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].scatter(y_test, y_pred, alpha=0.7)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        axs[0].plot([min_val, max_val], [min_val, max_val], linestyle="--")
        axs[0].set_title("Valeurs réelles vs prédictions")
        axs[0].set_xlabel("Valeurs réelles")
        axs[0].set_ylabel("Prédictions")
        axs[0].grid(True, linestyle="--", alpha=0.5)

        axs[1].plot(model.loss_curve_)
        axs[1].set_title("Courbe d'apprentissage")
        axs[1].set_xlabel("Itération")
        axs[1].set_ylabel("Perte")
        axs[1].grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=frame_graph)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        texte_resultat.config(
            text=(
                "Résultats du réseau de neurones\n\n"
                f"Architecture : ({hidden},)\n"
                f"Nombre d'itérations : {model.n_iter_}\n\n"
                f"R² : {r2:.4f}\n"
                f"MSE : {mse:.4f}\n"
                f"RMSE : {rmse:.4f}"
            )
        )

    except ValueError as e:
        messagebox.showerror("Erreur", str(e))
    except Exception as e:
        messagebox.showerror("Erreur", f"Une erreur est survenue :\n{e}")


root = tk.Tk()
root.title("Réseau de Neurones MLP")
root.geometry("1150x650")
root.configure(bg="#2c3e50")

frame_left = tk.Frame(root, bg="#2c3e50", width=270)
frame_left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

frame_graph = tk.Frame(root, bg="white")
frame_graph.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)


def ajouter_champ(label, valeur):
    tk.Label(frame_left, text=label, bg="#2c3e50", fg="white").pack(anchor="w")
    entry = tk.Entry(frame_left, width=12)
    entry.pack(pady=2)
    entry.insert(0, valeur)
    return entry


tk.Label(
    frame_left,
    text="Paramètres du réseau MLP",
    bg="#2c3e50",
    fg="white",
    font=("Arial", 11, "bold")
).pack(pady=10)

tk.Label(frame_left, text="Variable X1", bg="#2c3e50", fg="white", font=("Arial", 10, "bold")).pack(pady=(5, 2))
entry_x1_min = ajouter_champ("Minimum :", "-10")
entry_x1_max = ajouter_champ("Maximum :", "10")

tk.Label(frame_left, text="Variable X2", bg="#2c3e50", fg="white", font=("Arial", 10, "bold")).pack(pady=(10, 2))
entry_x2_min = ajouter_champ("Minimum :", "-5")
entry_x2_max = ajouter_champ("Maximum :", "15")

tk.Label(frame_left, text="Paramètres du modèle", bg="#2c3e50", fg="white", font=("Arial", 10, "bold")).pack(pady=(10, 2))
entry_n = ajouter_champ("Taille échantillon :", "300")
entry_hidden = ajouter_champ("Neurones cachés :", "50")
entry_iterations = ajouter_champ("Itérations max :", "800")

tk.Button(
    frame_left,
    text="Lancer le réseau MLP",
    command=lancer_reseau_neurones,
    bg="#1abc9c",
    fg="white",
    font=("Arial", 10, "bold"),
    width=22
).pack(pady=15)

texte_resultat = tk.Label(
    frame_left,
    text="Résultats ici...",
    bg="#2c3e50",
    fg="white",
    justify="left",
    anchor="w",
    font=("Arial", 10)
)
texte_resultat.pack(fill=tk.X, pady=10)

root.mainloop()
