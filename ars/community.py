import pandas as pd
import networkx as nx
# Lire les premières 6000 lignes du fichier texte dans un DataFrame pandas
df = pd.read_csv("stackoverflow.txt", sep=" ", nrows=6000)  

# Créer un nouveau graphe vide
G = nx.Graph()

for _, row in df.iterrows():
    values = row.values.tolist()
    if len(values) < 3:
        continue
    source = str(values[0])
    target = str(values[1])
    weight = int(values[2])
    G.add_edge(source, target, weight=weight)

# Appliquer l'algorithme k-clique
k_clique_communities = list(nx.algorithms.community.k_clique_communities(G, 3))  # Par exemple, chercher les cliques de taille 3

# Appliquer l'algorithme Louvain
louvain_communities = list(nx.community.greedy_modularity_communities(G))

# Appliquer l'algorithme démon/angel
demon_angel_communities = list(nx.community.label_propagation_communities(G))

# Afficher les résultats
print("Résultats de l'algorithme k-clique:")
print(k_clique_communities)
print("\nRésultats de l'algorithme Louvain:")
print(louvain_communities)
print("\nRésultats de l'algorithme démon/angel:")
print(demon_angel_communities)
