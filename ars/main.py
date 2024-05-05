import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import warnings

# Lire les premières 70000 lignes du fichier texte dans un DataFrame pandas
df = pd.read_csv("stackoverflow.txt", sep=" ", nrows=1000)  

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

# Affichage du graphe
def afficher_graphe():
    # Définir la disposition des nœuds du graphe
    pos = nx.spring_layout(G, k=0.15, iterations=20)

    # Dessiner le graphe avec la nouvelle disposition
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_size=100, font_size=10, node_color='skyblue', edge_color='gray', width=0.5)
    plt.title("Graphe Stack Overflow")
    plt.show()

# Affichage du diagramme de degré
def afficher_diagramme():
    degrees = [degree for node, degree in G.degree()]
    plt.hist(degrees, bins=100, color='skyblue', edgecolor='gray')
    plt.xlabel('Degré')
    plt.ylabel('Fréquence')
    plt.title('Distribution des degrés')
    plt.grid(True)
    plt.show()

# Affichage des composants connectés
def afficher_composants_connectes():
    composants = list(nx.connected_components(G))
    print(f"Il y a {len(composants)} composants connectés dans le graphe.")

# Affichage des chemins les plus courts
def afficher_chemins():
    chemins = dict(nx.all_pairs_shortest_path(G))
    for source, paths in chemins.items():
        for target, path in paths.items():
            print(f"Chemin le plus court de {source} à {target}: {path}")

# Affichage du coefficient de clustering
def afficher_coefficient_clustering():
    coefficient = nx.average_clustering(G)
    print(f"Le coefficient de clustering moyen est {coefficient}.")

# Affichage de l'analyse de la densité
def afficher_densite():
    densite = nx.density(G)
    print(f"La densité du graphe est {densite}.")

# Affichage de l'analyse de la centralité
def afficher_centralite():
    centralite = nx.degree_centrality(G)
    print(f"La centralité des nœuds a été calculée= {centralite}.")

# Appels aux différentes fonctions pour afficher les résultats
afficher_graphe()
afficher_diagramme()
afficher_composants_connectes()
afficher_chemins()
afficher_coefficient_clustering()
afficher_densite()
afficher_centralite()
