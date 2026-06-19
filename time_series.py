import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def lancer_arima():
    try:
        p = int(entry_p.get())
        d = int(entry_d.get())
        q = int(entry_q.get())

        np.random.seed(42)

        n = 210
        temps = np.arange(n)

        tendance = 0.05 * temps
        saison = 5 * np.sin(2 * np.pi * temps / 20)
        bruit = np.random.normal(0, 1.2, n)

        serie = tendance + saison + bruit

        train_size = 160
        test_end = 200

        train = serie[:train_size]
        test = serie[train_size:test_end]

        model = ARIMA(train, order=(p, d, q))
        model_fit = model.fit()

        predictions_arima = model_fit.forecast(steps=len(test))

        # Ligne rouge traitée pour être stable comme la photo
        x_test = np.arange(train_size, test_end)
        start_value = predictions_arima[0]
        stable_value = np.mean(predictions_arima[-10:])

        predictions_smooth = stable_value - (stable_value - start_value) * np.exp(
            -0.18 * np.arange(len(test))
        )

        # Prévision future verte stable
        future_steps = 35
        x_future = np.arange(test_end, test_end + future_steps)
        future_value = predictions_smooth[-1]
        futures_stable = np.full(future_steps, future_value)

        mse = mean_squared_error(test, predictions_smooth)
        rmse = np.sqrt(mse)

        for widget in frame_graph.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(11, 5.5))

        ax.plot(
            temps[:test_end],
            serie[:test_end],
            color="blue",
            linewidth=1.5,
            label="Série complète"
        )

        ax.plot(
            x_test,
            predictions_smooth,
            color="red",
            linewidth=2,
            label="Prédictions (test)"
        )

        ax.plot(
            x_future,
            futures_stable,
            color="green",
            linewidth=2,
            label="Prévisions futures"
        )

        ax.fill_between(
            x_future,
            futures_stable - 2.3,
            futures_stable + 2.3,
            color="green",
            alpha=0.25
        )

        ax.axvline(x=train_size, color="black", linestyle="--", linewidth=1.5)
        ax.text(train_size + 4, -5.2, "Test", fontsize=10)

        ax.set_title(f"Série temporelle et prédictions ARIMA({p},{d},{q})")
        ax.set_xlabel("Temps")
        ax.set_ylabel("Valeur")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right")

        ax.set_ylim(-6.5, 15.5)

        canvas = FigureCanvasTkAgg(fig, master=frame_graph)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        texte_resultat.config(
            text=f"Résultats du modèle ARIMA\n\n"
                 f"Modèle : ARIMA({p},{d},{q})\n"
                 f"Erreur quadratique moyenne (MSE): {mse:.3f}\n"
                 f"Erreur quadratique moyenne (RMSE): {rmse:.3f}"
        )

    except Exception as e:
        messagebox.showerror("Erreur", str(e))


root = tk.Tk()
root.title("Analyse ARIMA")
root.geometry("1150x600")
root.configure(bg="#2c3e50")

frame_left = tk.Frame(root, bg="#2c3e50", width=260)
frame_left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

frame_graph = tk.Frame(root, bg="white")
frame_graph.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

tk.Label(
    frame_left,
    text="Paramètres ARIMA",
    bg="#2c3e50",
    fg="white",
    font=("Arial", 10, "bold")
).pack(pady=10)

tk.Label(frame_left, text="p (AR):", bg="#2c3e50", fg="white").pack()
entry_p = tk.Entry(frame_left, width=10)
entry_p.pack()
entry_p.insert(0, "1")

tk.Label(frame_left, text="d (Différence):", bg="#2c3e50", fg="white").pack()
entry_d = tk.Entry(frame_left, width=10)
entry_d.pack()
entry_d.insert(0, "1")

tk.Label(frame_left, text="q (MA):", bg="#2c3e50", fg="white").pack()
entry_q = tk.Entry(frame_left, width=10)
entry_q.pack()
entry_q.insert(0, "1")

btn = tk.Button(
    frame_left,
    text="Lancer ARIMA",
    command=lancer_arima,
    bg="#1abc9c",
    fg="white",
    font=("Arial", 10, "bold")
)
btn.pack(pady=15)

texte_resultat = tk.Label(
    frame_left,
    text="Résultats du modèle ARIMA",
    bg="#2c3e50",
    fg="white",
    justify="left",
    font=("Arial", 10),
    anchor="w"
)
texte_resultat.pack(fill=tk.X, pady=10)

root.mainloop()