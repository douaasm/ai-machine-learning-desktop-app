import tkinter as tk
from tkinter import messagebox
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D


def lancer_regression():
    try:
        min_x1 = float(entry_x1_min.get())
        max_x1 = float(entry_x1_max.get())
        min_x2 = float(entry_x2_min.get())
        max_x2 = float(entry_x2_max.get())

        n = 200

        # Génération des données
        X1 = np.random.uniform(min_x1, max_x1, n)
        X2 = np.random.uniform(min_x2, max_x2, n)

        # Output Y
        Y = 4.5 + 2.0 * X1 - 1.4 * X2 + np.random.normal(0, 3, n)

        X = np.column_stack((X1, X2))

        # Modèle
        model = LinearRegression()
        model.fit(X, Y)

        Y_pred = model.predict(X)

        r2 = r2_score(Y, Y_pred)
        mse = mean_squared_error(Y, Y_pred)
        rmse = np.sqrt(mse)

        b0 = model.intercept_
        b1 = model.coef_[0]
        b2 = model.coef_[1]

        texte_resultat.config(
            text=f"Métriques de performance :\n"
                 f"R² = {r2:.4f}\n"
                 f"MSE = {mse:.4f}\n"
                 f"RMSE = {rmse:.4f}\n\n"
                 f"Coefficients du modèle :\n"
                 f"β0 = {b0:.4f}\n"
                 f"β1 = {b1:.4f}\n"
                 f"β2 = {b2:.4f}\n\n"
                 f"Équation du modèle :\n"
                 f"Y = {b0:.4f} + {b1:.4f}*X1 + {b2:.4f}*X2"
        )

        # Supprimer ancien graphe
        for widget in frame_graph.winfo_children():
            widget.destroy()

        # Figure 3D
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111, projection="3d")

        scatter = ax.scatter(
            X1, X2, Y,
            c=Y_pred,
            cmap="viridis",
            marker="o"
        )

        ax.set_title("Régression Linéaire Multiple en 3D")
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_zlabel("Y")

        colorbar = fig.colorbar(scatter, ax=ax, shrink=0.7)
        colorbar.set_label("Valeur Y prédite")

        ax.view_init(elev=25, azim=-60)

        canvas = FigureCanvasTkAgg(fig, master=frame_graph)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    except ValueError:
        messagebox.showerror("Erreur", "Veuillez entrer des valeurs valides.")


# Fenêtre principale
root = tk.Tk()
root.title("Régression Linéaire Multiple en 3D")
root.geometry("1000x600")
root.configure(bg="#2c3e50")

# Frame gauche
frame_left = tk.Frame(root, bg="#2c3e50", width=230)
frame_left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

# Frame graphique
frame_graph = tk.Frame(root, bg="white")
frame_graph.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

# Paramètres
tk.Label(
    frame_left,
    text="Paramètres des données",
    bg="#2c3e50",
    fg="white",
    font=("Arial", 10, "bold")
).pack(pady=10)

# Variable X1
tk.Label(frame_left, text="Variable X1", bg="#2c3e50", fg="white").pack()
tk.Label(frame_left, text="Minimum :", bg="#2c3e50", fg="white").pack()
entry_x1_min = tk.Entry(frame_left)
entry_x1_min.pack()
entry_x1_min.insert(0, "-10")

tk.Label(frame_left, text="Maximum :", bg="#2c3e50", fg="white").pack()
entry_x1_max = tk.Entry(frame_left)
entry_x1_max.pack()
entry_x1_max.insert(0, "10")

# Variable X2
tk.Label(frame_left, text="Variable X2", bg="#2c3e50", fg="white").pack(pady=(15, 0))
tk.Label(frame_left, text="Minimum :", bg="#2c3e50", fg="white").pack()
entry_x2_min = tk.Entry(frame_left)
entry_x2_min.pack()
entry_x2_min.insert(0, "-5")

tk.Label(frame_left, text="Maximum :", bg="#2c3e50", fg="white").pack()
entry_x2_max = tk.Entry(frame_left)
entry_x2_max.pack()
entry_x2_max.insert(0, "15")

btn = tk.Button(
    frame_left,
    text="Lancer Régression",
    command=lancer_regression,
    bg="#1abc9c",
    fg="white",
    font=("Arial", 10, "bold")
)
btn.pack(pady=15)

texte_resultat = tk.Label(
    frame_left,
    text="Résultats ici...",
    bg="#2c3e50",
    fg="white",
    justify="left",
    font=("Consolas", 9)
)
texte_resultat.pack(pady=10)

root.mainloop()