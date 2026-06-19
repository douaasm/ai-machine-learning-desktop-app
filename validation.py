import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, f1_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def lancer_validation():
    try:
        folds = int(entry_folds.get())

        X, y = make_classification(
            n_samples=300,
            n_features=3,
            n_informative=3,
            n_redundant=0,
            n_classes=3,
            random_state=42
        )

        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Réseau de Neurones": MLPClassifier(hidden_layer_sizes=(50,), max_iter=800, random_state=42),
            "SVM": SVC(kernel="rbf", random_state=42),
            "Arbre de Décision": DecisionTreeClassifier(random_state=42)
        }

        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

        noms = []
        acc_means = []
        acc_stds = []
        f1_means = []
        f1_stds = []
        times = []
        fold_scores = {}

        texte = "Résultats de la validation croisée\n\n"

        for name, model in models.items():
            start = time.time()

            acc_scores = cross_val_score(
                model, X, y, cv=cv, scoring=make_scorer(accuracy_score)
            )

            f1_scores = cross_val_score(
                model, X, y, cv=cv, scoring=make_scorer(f1_score, average="macro")
            )

            # Temps moyen par fold
            duration = (time.time() - start) / folds

            noms.append(name)
            acc_means.append(acc_scores.mean())
            acc_stds.append(acc_scores.std())
            f1_means.append(f1_scores.mean())
            f1_stds.append(f1_scores.std())
            times.append(duration)
            fold_scores[name] = acc_scores

            texte += f"{name}:\n\n"
            texte += f"    Exactitude moyenne: {acc_scores.mean():.4f} ± {acc_scores.std():.4f}\n"
            texte += f"    Score F1 moyen: {f1_scores.mean():.4f} ± {f1_scores.std():.4f}\n"
            texte += f"    Temps d'entraînement moyen: {duration:.2f}s\n\n"

        # 🔥 Trouver le meilleur modèle
        best_model = noms[np.argmax(acc_means)]
        texte += f"\nMeilleur modèle: {best_model}"

        texte_resultat.config(text=texte)

        for widget in frame_graph.winfo_children():
            widget.destroy()

        # Style pro
        plt.style.use('seaborn-v0_8')

        fig, axs = plt.subplots(2, 2, figsize=(10, 7))

        colors = ["#3498db", "#e74c3c", "#2ecc71", "#f1c40f"]

        # 1️⃣ Exactitudes
        axs[0, 0].bar(noms, acc_means, yerr=acc_stds, capsize=5, color=colors)
        axs[0, 0].set_title("Comparaison des exactitudes")
        axs[0, 0].set_ylim(0, 1)
        axs[0, 0].tick_params(axis="x", rotation=45)

        # 2️⃣ F1
        axs[0, 1].bar(noms, f1_means, yerr=f1_stds, capsize=5, color=colors)
        axs[0, 1].set_title("Comparaison des scores F1")
        axs[0, 1].set_ylim(0, 1)
        axs[0, 1].tick_params(axis="x", rotation=45)

        # 3️⃣ Temps (corrigé visuellement)
        axs[1, 0].bar(noms, times, color=colors)
        axs[1, 0].set_title("Comparaison des temps d'entraînement")
        axs[1, 0].set_ylim(0, max(times) * 1.2)
        axs[1, 0].tick_params(axis="x", rotation=45)

        # 4️⃣ Evolution par fold
        folds_range = range(1, folds + 1)
        for name, scores in fold_scores.items():
            axs[1, 1].plot(folds_range, scores, marker="o", label=name)

        axs[1, 1].set_title("Évolution des performances par fold")
        axs[1, 1].set_xlabel("Fold")
        axs[1, 1].set_ylabel("Exactitude")
        axs[1, 1].grid(True)
        axs[1, 1].legend()

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=frame_graph)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    except Exception as e:
        messagebox.showerror("Erreur", str(e))


# Interface
root = tk.Tk()
root.title("Validation Croisée")
root.geometry("1200x700")
root.configure(bg="#2c3e50")

frame_left = tk.Frame(root, bg="#2c3e50", width=270)
frame_left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

frame_graph = tk.Frame(root, bg="white")
frame_graph.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

tk.Label(
    frame_left,
    text="Paramètres de Validation",
    bg="#2c3e50",
    fg="white",
    font=("Arial", 10, "bold")
).pack(pady=10)

tk.Label(frame_left, text="Nombre de folds:", bg="#2c3e50", fg="white").pack()

entry_folds = tk.Entry(frame_left)
entry_folds.pack()
entry_folds.insert(0, "5")

btn = tk.Button(
    frame_left,
    text="Lancer Validation",
    command=lancer_validation,
    bg="#1abc9c",
    fg="white",
    font=("Arial", 10, "bold")
)
btn.pack(pady=15)

texte_resultat = tk.Label(
    frame_left,
    text="Résultats de la validation croisée",
    bg="#2c3e50",
    fg="white",
    justify="left",
    anchor="w"
)
texte_resultat.pack(fill=tk.X, pady=10)

root.mainloop()