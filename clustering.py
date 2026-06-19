import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def lancer_clustering():
    try:
        min_x1 = float(entry_x1_min.get())
        max_x1 = float(entry_x1_max.get())
        min_x2 = float(entry_x2_min.get())
        max_x2 = float(entry_x2_max.get())
        min_x3 = float(entry_x3_min.get())
        max_x3 = float(entry_x3_max.get())

        k = int(entry_k.get())
        n = int(entry_n.get())

        centers_init = [
            [-5, 0, 5],
            [1, 5, 10],
            [5, 10, 15]
        ]

        X, _ = make_blobs(
            n_samples=n,
            centers=centers_init,
            cluster_std=1.4,
            random_state=42
        )

        X[:, 0] = np.clip(X[:, 0], min_x1, max_x1)
        X[:, 1] = np.clip(X[:, 1], min_x2, max_x2)
        X[:, 2] = np.clip(X[:, 2], min_x3, max_x3)

        X1 = X[:, 0]
        X2 = X[:, 1]
        X3 = X[:, 2]

        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(X)

        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        score = silhouette_score(X, labels)

        for widget in frame_graph.winfo_children():
            widget.destroy()

        fig, axs = plt.subplots(2, 2, figsize=(9, 6))

        axs[0, 0].scatter(X1, X2, c=labels, cmap="viridis", s=25)
        axs[0, 0].scatter(centers[:, 0], centers[:, 1], c="red", s=160, marker="*")
        axs[0, 0].set_title("Clustering: X1 vs X2")
        axs[0, 0].set_xlabel("X1")
        axs[0, 0].set_ylabel("X2")
        axs[0, 0].grid(True, linestyle="--", alpha=0.5)

        axs[0, 1].scatter(X1, X3, c=labels, cmap="viridis", s=25)
        axs[0, 1].scatter(centers[:, 0], centers[:, 2], c="red", s=160, marker="*")
        axs[0, 1].set_title("Clustering: X1 vs X3")
        axs[0, 1].set_xlabel("X1")
        axs[0, 1].set_ylabel("X3")
        axs[0, 1].grid(True, linestyle="--", alpha=0.5)

        axs[1, 0].scatter(X2, X3, c=labels, cmap="viridis", s=25)
        axs[1, 0].scatter(centers[:, 1], centers[:, 2], c="red", s=160, marker="*")
        axs[1, 0].set_title("Clustering: X2 vs X3")
        axs[1, 0].set_xlabel("X2")
        axs[1, 0].set_ylabel("X3")
        axs[1, 0].grid(True, linestyle="--", alpha=0.5)

        axs[1, 1].axis("off")

        handles, labels_legend = axs[0, 0].get_legend_handles_labels()
        axs[1, 1].legend(
            *axs[0, 0].scatter(X1, X2, c=labels, cmap="viridis").legend_elements(),
            title="Clusters",
            loc="center"
        )

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=frame_graph)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        texte = f"Résultats du clustering\n\n"
        texte += f"Nombre de clusters : {k}\n"
        texte += f"Score de silhouette moyen : {score:.4f}\n\n"
        texte += "Centres des clusters\n"
        for i, c in enumerate(centers):
            texte += f"Cluster {i+1}: [{c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}]\n"

        texte_resultat.config(text=texte)

    except ValueError:
        messagebox.showerror("Erreur", "Valeurs invalides")


root = tk.Tk()
root.title("Clustering K-Means")
root.geometry("1100x600")
root.configure(bg="#2c3e50")

frame_left = tk.Frame(root, bg="#2c3e50", width=260)
frame_left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

frame_graph = tk.Frame(root, bg="white")
frame_graph.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

tk.Label(frame_left, text="Paramètres du Clustering", bg="#2c3e50", fg="white", font=("Arial", 10, "bold")).pack(pady=5)

tk.Label(frame_left, text="Variable X1", bg="#2c3e50", fg="white").pack()
entry_x1_min = tk.Entry(frame_left, width=10)
entry_x1_min.pack()
entry_x1_min.insert(0, "-10")
entry_x1_max = tk.Entry(frame_left, width=10)
entry_x1_max.pack()
entry_x1_max.insert(0, "10")

tk.Label(frame_left, text="Variable X2", bg="#2c3e50", fg="white").pack()
entry_x2_min = tk.Entry(frame_left, width=10)
entry_x2_min.pack()
entry_x2_min.insert(0, "-5")
entry_x2_max = tk.Entry(frame_left, width=10)
entry_x2_max.pack()
entry_x2_max.insert(0, "15")

tk.Label(frame_left, text="Variable X3", bg="#2c3e50", fg="white").pack()
entry_x3_min = tk.Entry(frame_left, width=10)
entry_x3_min.pack()
entry_x3_min.insert(0, "0")
entry_x3_max = tk.Entry(frame_left, width=10)
entry_x3_max.pack()
entry_x3_max.insert(0, "20")

tk.Label(frame_left, text="Paramètres du modèle", bg="#2c3e50", fg="white", font=("Arial", 10, "bold")).pack(pady=8)

tk.Label(frame_left, text="Nombre de clusters", bg="#2c3e50", fg="white").pack()
entry_k = tk.Entry(frame_left, width=10)
entry_k.pack()
entry_k.insert(0, "3")

tk.Label(frame_left, text="Taille échantillon", bg="#2c3e50", fg="white").pack()
entry_n = tk.Entry(frame_left, width=10)
entry_n.pack()
entry_n.insert(0, "300")

btn = tk.Button(frame_left, text="Lancer Clustering", command=lancer_clustering, bg="#1abc9c", fg="white")
btn.pack(pady=10)

texte_resultat = tk.Label(frame_left, text="Résultats ici...", bg="#2c3e50", fg="white", justify="left", font=("Consolas", 9))
texte_resultat.pack(pady=10)

root.mainloop()