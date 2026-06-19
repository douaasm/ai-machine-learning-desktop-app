import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score

from statsmodels.tsa.arima.model import ARIMA

data = None


def importer_csv():
    global data

    fichier = filedialog.askopenfilename(filetypes=[("Fichiers CSV", "*.csv")])

    if fichier:
        try:
            data = pd.read_csv(fichier)

            label_fichier_reg.config(text=f"Fichier chargé : {fichier}")
            label_fichier_rf.config(text=f"Fichier chargé : {fichier}")
            label_fichier_clustering.config(text=f"Fichier chargé : {fichier}")
            label_fichier_arima.config(text=f"Fichier chargé : {fichier}")
            label_fichier_val.config(text=f"Fichier chargé : {fichier}")

            afficher_tableau()
            charger_colonnes()

            messagebox.showinfo("Succès", "CSV chargé avec succès")

        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de lire le fichier :\n{e}")


def afficher_tableau():
    for i in tree.get_children():
        tree.delete(i)

    tree["columns"] = list(data.columns)
    tree["show"] = "headings"

    for col in data.columns:
        tree.heading(col, text=col)
        tree.column(col, width=120)

    for _, row in data.iterrows():
        tree.insert("", "end", values=list(row))


def charger_colonnes():
    colonnes = list(data.columns)

    combo_x_reg["values"] = colonnes
    combo_y_reg["values"] = colonnes

    combo_x_rf["values"] = colonnes
    combo_y_rf["values"] = colonnes

    combo_x_cluster["values"] = colonnes
    combo_y_cluster["values"] = colonnes

    combo_arima["values"] = colonnes

    combo_x_val["values"] = colonnes
    combo_y_val["values"] = colonnes


def entrainer_regression():
    if data is None:
        messagebox.showerror("Erreur", "Importer un fichier CSV")
        return

    if combo_x_reg.get() == "" or combo_y_reg.get() == "":
        messagebox.showerror("Erreur", "Sélectionner X et y")
        return

    X = data[[combo_x_reg.get()]]
    y = data[combo_y_reg.get()]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)

    messagebox.showinfo("Régression", f"Score R² : {round(score, 3)}")

    plt.figure()
    plt.scatter(X, y, label="Données réelles")
    plt.plot(X, model.predict(X), linestyle="--", label="Modèle")
    plt.xlabel(combo_x_reg.get())
    plt.ylabel(combo_y_reg.get())
    plt.title("Régression linéaire")
    plt.legend()
    plt.grid()
    plt.show()


def entrainer_random_forest():
    if data is None:
        messagebox.showerror("Erreur", "Importer un fichier CSV")
        return

    if combo_x_rf.get() == "" or combo_y_rf.get() == "":
        messagebox.showerror("Erreur", "Sélectionner X et y")
        return

    X = data[[combo_x_rf.get()]]
    y = data[combo_y_rf.get()]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)

    messagebox.showinfo("Random Forest", f"Score R² : {round(score, 3)}")

    plt.figure()
    plt.scatter(X, y, label="Données réelles")
    plt.scatter(X_test, y_pred, marker="x", label="Prédictions")
    plt.xlabel(combo_x_rf.get())
    plt.ylabel(combo_y_rf.get())
    plt.title("Random Forest")
    plt.legend()
    plt.grid()
    plt.show()


def lancer_clustering():
    if data is None:
        messagebox.showerror("Erreur", "Importer un fichier CSV")
        return

    x_col = combo_x_cluster.get()
    y_col = combo_y_cluster.get()

    if x_col == "" or y_col == "":
        messagebox.showerror("Erreur", "Sélectionner X et Y")
        return

    try:
        k = int(entry_k.get())
    except:
        messagebox.showerror("Erreur", "Nombre de clusters invalide")
        return

    X = data[[x_col, y_col]]

    model = KMeans(n_clusters=k, random_state=42)
    clusters = model.fit_predict(X)

    messagebox.showinfo("Clustering", "Clustering terminé")

    plt.figure()
    plt.scatter(data[x_col], data[y_col], c=clusters)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title("Clustering KMeans")
    plt.grid()
    plt.show()


def lancer_arima():
    if data is None:
        messagebox.showerror("Erreur", "Importer un fichier CSV")
        return

    colonne = combo_arima.get()

    if colonne == "":
        messagebox.showerror("Erreur", "Sélectionner une colonne")
        return

    try:
        p = int(entry_p.get())
        d = int(entry_d.get())
        q = int(entry_q.get())
        steps = int(entry_steps.get())
    except:
        messagebox.showerror("Erreur", "Valeurs ARIMA invalides")
        return

    serie = pd.to_numeric(data[colonne], errors="coerce").dropna()

    try:
        model = ARIMA(serie, order=(p, d, q))
        result = model.fit()
        forecast = result.forecast(steps=steps)

        messagebox.showinfo("ARIMA", "Prévision terminée")

        plt.figure()
        plt.plot(serie.values, label="Données réelles")
        plt.plot(
            range(len(serie), len(serie) + steps),
            forecast,
            marker="o",
            label="Prévision"
        )
        plt.xlabel("Temps")
        plt.ylabel(colonne)
        plt.title("Prévision ARIMA")
        plt.legend()
        plt.grid()
        plt.show()

    except Exception as e:
        messagebox.showerror("Erreur ARIMA", str(e))


def validation_croisee():
    if data is None:
        messagebox.showerror("Erreur", "Importer un fichier CSV")
        return

    if combo_x_val.get() == "" or combo_y_val.get() == "":
        messagebox.showerror("Erreur", "Sélectionner X et y")
        return

    X = data[[combo_x_val.get()]]
    y = data[combo_y_val.get()]

    n = len(data)

    if n < 4:
        messagebox.showerror("Erreur", "Il faut au moins 4 lignes pour la validation croisée")
        return

    if n < 10:
        cv = 3
    else:
        cv = 5

    model = LinearRegression()

    try:
        scores = cross_val_score(model, X, y, cv=cv, scoring="r2")

        messagebox.showinfo(
            "Validation croisée",
            f"CV = {cv}\n\nScores R² : {scores}\n\nMoyenne : {round(scores.mean(), 3)}"
        )

        plt.figure()
        plt.plot(range(1, len(scores) + 1), scores, marker="o")
        plt.xlabel("Fold")
        plt.ylabel("Score R²")
        plt.title("Validation croisée")
        plt.grid()
        plt.show()

    except Exception as e:
        messagebox.showerror("Erreur validation croisée", str(e))


fenetre = tk.Tk()
fenetre.title("Application IA - Machine Learning")
fenetre.geometry("1050x700")

titre = tk.Label(
    fenetre,
    text="Application IA - Machine Learning",
    font=("Arial", 22, "bold")
)
titre.pack(pady=15)

notebook = ttk.Notebook(fenetre)
notebook.pack(expand=True, fill="both", padx=10, pady=10)

tab_regression = ttk.Frame(notebook)
tab_random_forest = ttk.Frame(notebook)
tab_clustering = ttk.Frame(notebook)
tab_arima = ttk.Frame(notebook)
tab_validation = ttk.Frame(notebook)

notebook.add(tab_regression, text="Régression")
notebook.add(tab_random_forest, text="Random Forest")
notebook.add(tab_clustering, text="Clustering")
notebook.add(tab_arima, text="ARIMA")
notebook.add(tab_validation, text="Validation croisée")


ttk.Label(tab_regression, text="Module Régression", font=("Arial", 16)).pack(pady=10)

ttk.Button(tab_regression, text="Importer CSV", command=importer_csv).pack(pady=5)
label_fichier_reg = ttk.Label(tab_regression, text="Aucun fichier chargé")
label_fichier_reg.pack()

ttk.Label(tab_regression, text="Variable X").pack()
combo_x_reg = ttk.Combobox(tab_regression)
combo_x_reg.pack()

ttk.Label(tab_regression, text="Variable y").pack()
combo_y_reg = ttk.Combobox(tab_regression)
combo_y_reg.pack()

ttk.Button(tab_regression, text="Entrainer Régression", command=entrainer_regression).pack(pady=10)

frame_table = ttk.Frame(tab_regression)
frame_table.pack(fill="both", expand=True)

tree = ttk.Treeview(frame_table)
tree.pack(side="left", fill="both", expand=True)

scrollbar = ttk.Scrollbar(frame_table, command=tree.yview)
scrollbar.pack(side="right", fill="y")
tree.configure(yscrollcommand=scrollbar.set)


ttk.Label(tab_random_forest, text="Module Random Forest", font=("Arial", 16)).pack(pady=10)

ttk.Button(tab_random_forest, text="Importer CSV", command=importer_csv).pack(pady=5)
label_fichier_rf = ttk.Label(tab_random_forest, text="Aucun fichier chargé")
label_fichier_rf.pack()

ttk.Label(tab_random_forest, text="Variable X").pack()
combo_x_rf = ttk.Combobox(tab_random_forest)
combo_x_rf.pack()

ttk.Label(tab_random_forest, text="Variable y").pack()
combo_y_rf = ttk.Combobox(tab_random_forest)
combo_y_rf.pack()

ttk.Button(tab_random_forest, text="Entrainer Random Forest", command=entrainer_random_forest).pack(pady=10)


ttk.Label(tab_clustering, text="Module Clustering KMeans", font=("Arial", 16)).pack(pady=10)

ttk.Button(tab_clustering, text="Importer CSV", command=importer_csv).pack(pady=5)
label_fichier_clustering = ttk.Label(tab_clustering, text="Aucun fichier chargé")
label_fichier_clustering.pack()

ttk.Label(tab_clustering, text="Variable X").pack()
combo_x_cluster = ttk.Combobox(tab_clustering)
combo_x_cluster.pack()

ttk.Label(tab_clustering, text="Variable Y").pack()
combo_y_cluster = ttk.Combobox(tab_clustering)
combo_y_cluster.pack()

ttk.Label(tab_clustering, text="Nombre de clusters").pack()
entry_k = ttk.Entry(tab_clustering)
entry_k.insert(0, "2")
entry_k.pack()

ttk.Button(tab_clustering, text="Lancer Clustering", command=lancer_clustering).pack(pady=10)


ttk.Label(tab_arima, text="Module ARIMA", font=("Arial", 16)).pack(pady=10)

ttk.Button(tab_arima, text="Importer CSV", command=importer_csv).pack(pady=5)
label_fichier_arima = ttk.Label(tab_arima, text="Aucun fichier chargé")
label_fichier_arima.pack()

ttk.Label(tab_arima, text="Colonne à prévoir").pack()
combo_arima = ttk.Combobox(tab_arima)
combo_arima.pack()

ttk.Label(tab_arima, text="p").pack()
entry_p = ttk.Entry(tab_arima)
entry_p.insert(0, "1")
entry_p.pack()

ttk.Label(tab_arima, text="d").pack()
entry_d = ttk.Entry(tab_arima)
entry_d.insert(0, "1")
entry_d.pack()

ttk.Label(tab_arima, text="q").pack()
entry_q = ttk.Entry(tab_arima)
entry_q.insert(0, "0")
entry_q.pack()

ttk.Label(tab_arima, text="Nombre de prévisions").pack()
entry_steps = ttk.Entry(tab_arima)
entry_steps.insert(0, "3")
entry_steps.pack()

ttk.Button(tab_arima, text="Lancer ARIMA", command=lancer_arima).pack(pady=10)


ttk.Label(tab_validation, text="Module Validation croisée", font=("Arial", 16)).pack(pady=10)

ttk.Button(tab_validation, text="Importer CSV", command=importer_csv).pack(pady=5)
label_fichier_val = ttk.Label(tab_validation, text="Aucun fichier chargé")
label_fichier_val.pack()

ttk.Label(tab_validation, text="Variable X").pack()
combo_x_val = ttk.Combobox(tab_validation)
combo_x_val.pack()

ttk.Label(tab_validation, text="Variable y").pack()
combo_y_val = ttk.Combobox(tab_validation)
combo_y_val.pack()

ttk.Button(
    tab_validation,
    text="Lancer Validation Croisée",
    command=validation_croisee
).pack(pady=10)


fenetre.mainloop()