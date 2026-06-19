import tkinter as tk
from tkinter import messagebox
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def ouvrir_script(nom_script):
    try:
        chemin_script = BASE_DIR / nom_script
        if not chemin_script.exists():
            raise FileNotFoundError(f"Fichier introuvable : {nom_script}")
        subprocess.Popen([sys.executable, str(chemin_script)])
    except Exception as e:
        messagebox.showerror("Erreur", str(e))


root = tk.Tk()
root.title("Application Intelligence Artificielle")
root.geometry("520x470")
root.configure(bg="#2c3e50")

tk.Label(
    root,
    text="Application IA - Menu Principal",
    font=("Arial", 16, "bold"),
    bg="#2c3e50",
    fg="white"
).pack(pady=20)


def bouton(text, script):
    return tk.Button(
        root,
        text=text,
        command=lambda: ouvrir_script(script),
        bg="#1abc9c",
        fg="white",
        font=("Arial", 12, "bold"),
        width=28,
        pady=5
    )


bouton("Régression Linéaire", "regression.py").pack(pady=5)
bouton("Clustering K-Means", "clustering.py").pack(pady=5)
bouton("Random Forest", "random_forest.py").pack(pady=5)
bouton("Time Series (ARIMA)", "time_series.py").pack(pady=5)
bouton("Réseau de Neurones MLP", "neural_network.py").pack(pady=5)
bouton("Validation Croisée", "validation.py").pack(pady=5)

tk.Button(
    root,
    text="Quitter",
    command=root.quit,
    bg="#e74c3c",
    fg="white",
    font=("Arial", 12, "bold"),
    width=28
).pack(pady=20)

root.mainloop()
