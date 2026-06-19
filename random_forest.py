import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def lancer_random_forest():
    try:
        min_x1 = float(entry_x1_min.get())
        max_x1 = float(entry_x1_max.get())
        min_x2 = float(entry_x2_min.get())
        max_x2 = float(entry_x2_max.get())
        min_x3 = float(entry_x3_min.get())
        max_x3 = float(entry_x3_max.get())

        n_trees = int(entry_trees.get())
        n_classes = int(entry_classes.get())

        # Données proches du TP
        centers = [
            [-7, 12, 5],
            [1, 5, 10],
            [8, 0, 15]
        ]

        X, y = make_blobs(
            n_samples=60,
            centers=centers,
            cluster_std=2.2,
            n_features=3,
            random_state=7
        )

        X[:, 0] = np.clip(X[:, 0], min_x1, max_x1)
        X[:, 1] = np.clip(X[:, 1], min_x2, max_x2)
        X[:, 2] = np.clip(X[:, 2], min_x3, max_x3)

        model = RandomForestClassifier(
            n_estimators=n_trees,
            random_state=42,
            max_depth=5
        )

        model.fit(X, y)
        y_pred = model.predict(X)

        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, average="macro")
        rec = recall_score(y, y_pred, average="macro")
        f1 = f1_score(y, y_pred, average="macro")
        cm = confusion_matrix(y, y_pred)

        importances = model.feature_importances_

        for widget in frame_graph.winfo_children():
            widget.destroy()

        fig = plt.figure(figsize=(11, 7))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.35])

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])

        # 1) Importance des variables
        ax1.bar(["X1", "X2", "X3"], importances)
        ax1.set_title("Importance des variables")
        ax1.set_ylabel("Importance")

        # 2) Matrice de confusion
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax2,
            xticklabels=["Classe A", "Classe B", "Classe C"],
            yticklabels=["Classe A", "Classe B", "Classe C"]
        )
        ax2.set_title("Matrice de confusion")
        ax2.set_xlabel("Prédictions")
        ax2.set_ylabel("Valeurs réelles")

        # 3) Frontières de décision
        x_min, x_max = min_x1, max_x1
        y_min, y_max = min_x2, max_x2

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 300),
            np.linspace(y_min, y_max, 300)
        )

        x3_mean = np.mean(X[:, 2])
        grid = np.c_[xx.ravel(), yy.ravel(), np.full(xx.ravel().shape, x3_mean)]

        Z = model.predict(grid)
        Z = Z.reshape(xx.shape)

        ax3.contourf(xx, yy, Z, alpha=0.35, cmap="viridis")
        scatter = ax3.scatter(
            X[:, 0],
            X[:, 1],
            c=y,
            cmap="viridis",
            edgecolor="black",
            s=35
        )

        ax3.set_title("Frontières de décision (avec X3 fixé à sa valeur moyenne)")
        ax3.set_xlabel("X1")
        ax3.set_ylabel("X2")
        ax3.set_xlim(min_x1, max_x1)
        ax3.set_ylim(min_x2, max_x2)

        legend = ax3.legend(*scatter.legend_elements(), title="Classes", loc="upper right")
        ax3.add_artist(legend)

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=frame_graph)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        texte_resultat.config(
            text=f"Métriques de performance\n\n"
                 f"Exactitude (Accuracy): {acc:.4f}\n"
                 f"Précision moyenne: {prec:.4f}\n"
                 f"Rappel moyen: {rec:.4f}\n"
                 f"Score F1 moyen: {f1:.4f}\n\n"
                 f"Importance des variables\n\n"
                 f"X1: {importances[0]:.4f} ({importances[0]*100:.2f}%)\n"
                 f"X2: {importances[1]:.4f} ({importances[1]*100:.2f}%)\n"
                 f"X3: {importances[2]:.4f} ({importances[2]*100:.2f}%)"
        )

    except ValueError:
        messagebox.showerror("Erreur", "Veuillez entrer des valeurs valides.")


root = tk.Tk()
root.title("Random Forest - Analyse")
root.geometry("1250x720")
root.configure(bg="#2c3e50")

frame_left = tk.Frame(root, bg="#2c3e50", width=270)
frame_left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

frame_graph = tk.Frame(root, bg="white")
frame_graph.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)


def section(title):
    lbl = tk.Label(
        frame_left,
        text=title,
        bg="#2c3e50",
        fg="white",
        font=("Arial", 10, "bold"),
        anchor="w"
    )
    lbl.pack(fill=tk.X, pady=(8, 2))


def champ(label, value):
    tk.Label(frame_left, text=label, bg="#2c3e50", fg="white").pack(anchor="w")
    entry = tk.Entry(frame_left, width=10)
    entry.pack(anchor="center")
    entry.insert(0, value)
    return entry


section("Paramètres Random Forest")

section("Variable X1")
entry_x1_min = champ("Minimum:", "-10")
entry_x1_max = champ("Maximum:", "10")

section("Variable X2")
entry_x2_min = champ("Minimum:", "-5")
entry_x2_max = champ("Maximum:", "15")

section("Variable X3")
entry_x3_min = champ("Minimum:", "0")
entry_x3_max = champ("Maximum:", "20")

section("Paramètres du modèle")
entry_trees = champ("Nombre d'arbres:", "100")
entry_classes = champ("Nombre de classes:", "3")

btn = tk.Button(
    frame_left,
    text="Lancer Random Forest",
    command=lancer_random_forest,
    bg="#0b8f3a",
    fg="white",
    font=("Arial", 10, "bold"),
    width=22
)
btn.pack(pady=15)

texte_resultat = tk.Label(
    frame_left,
    text="Métriques de performance\n\n",
    bg="#2c3e50",
    fg="white",
    justify="left",
    font=("Arial", 10),
    anchor="w"
)
texte_resultat.pack(fill=tk.X, pady=10)

root.mainloop()