#algorithme K-means
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Lecture du fichier texte et séparation des valeurs par des espaces
df = pd.read_csv("stackoverflow.txt", sep=" ", header=None)

# Calcul du nombre de questions publiées par chaque utilisateur
questions_publiees = df[df[0] == 9].groupby(1).size().reset_index(name='Nombre_questions_publiees')

# Calcul du nombre de réponses fournies par chaque utilisateur
reponses_fournies = df[df[0] == 1].groupby(1).size().reset_index(name='Nombre_reponses_fournies')

# Calcul du nombre de commentaires écrits par chaque utilisateur
commentaires_ecrits = df[df[0] == 2].groupby(1).size().reset_index(name='Nombre_commentaires_ecrits')

# Fusion des statistiques pour chaque utilisateur en un seul DataFrame
merged_df = questions_publiees.merge(reponses_fournies, on=1, how='outer').merge(commentaires_ecrits, on=1, how='outer')

# Remplacer les valeurs NaN par 0
merged_df = merged_df.fillna(0)

# Utiliser les caractéristiques pour le clustering
X = merged_df.drop(columns=[1])

# Application de l'algorithme K-Means
kmeans = KMeans(n_clusters=3) 
clusters = kmeans.fit_predict(X)

# Ajout des clusters au DataFrame
merged_df['Cluster'] = clusters

# Affichage du DataFrame avec les clusters
print(merged_df)

# Visualisation des clusters
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters, cmap='viridis')
plt.xlabel('Nombre de questions publiées')
plt.ylabel('Nombre de réponses fournies')
plt.title('Clustering des utilisateurs')
plt.colorbar(label='Cluster')
plt.show()
