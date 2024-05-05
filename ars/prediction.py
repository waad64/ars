import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Lecture du fichier texte et séparation des valeurs par des espaces
df = pd.read_csv("stackoverflow.txt", sep=" ", header=None, names=["Action", "Utilisateur", "Timestamp"])

# Calcul du nombre de questions publiées par chaque utilisateur
questions_publiees = df[df["Action"] == 9].groupby("Utilisateur").size().reset_index(name='Nombre_questions_publiees')

# Calcul du nombre de réponses fournies par chaque utilisateur
reponses_fournies = df[df["Action"] == 1].groupby("Utilisateur").size().reset_index(name='Nombre_reponses_fournies')

# Calcul du nombre de commentaires écrits par chaque utilisateur
commentaires_ecrits = df[df["Action"] == 2].groupby("Utilisateur").size().reset_index(name='Nombre_commentaires_ecrits')

# Fusion des statistiques pour chaque utilisateur en un seul DataFrame
merged_df = questions_publiees.merge(reponses_fournies, on="Utilisateur", how="outer")
merged_df = merged_df.merge(commentaires_ecrits, on="Utilisateur", how="outer")

# Suppression des lignes avec des valeurs manquantes
merged_df = merged_df.dropna()

# Conversion des données en types numériques pour le modèle SVM
X = merged_df.drop(columns=['Utilisateur'])
X = X.astype(int)  # Conversion en entiers
y = merged_df['Utilisateur']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialisation du modèle SVM
svm_model = SVC()

# Entraînement du modèle SVM
svm_model.fit(X_train, y_train)

# Évaluation du modèle sur l'ensemble de test
accuracy = svm_model.score(X_test, y_test)
print("Précision du modèle SVM :", accuracy)
