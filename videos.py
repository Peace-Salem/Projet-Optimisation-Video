#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
videos.py

Script principal pour le projet Hash Code 2017 - Streaming videos.
Lit un dataset, construit un modèle Gurobi, l'écrit en .mps, résout,
et produit un fichier videos.out au format attendu.

Usage (dans la racine du projet) :
    python videos.py data/example.in
    python videos.py data/trending_4000_10k.in
"""

import sys
from collections import defaultdict
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB


# =========================
#  Lecture et pré-traitement
# =========================

def parse_input(path):
    """
    Lit le fichier d'entrée HashCode et renvoie une structure de données Python.

    Format d'entrée :
        V E R C X
        size_0 size_1 ... size_{V-1}
        Pour chaque endpoint e :
            Ldc[e] K
            (K lignes) : cache_id latency
        Puis R lignes de requêtes :
            video_id endpoint_id n_requests

    Retourne un dict :
    {
        'V': int,
        'E': int,
        'R': int,
        'C': int,
        'X': int,
        'video_sizes': [int] de taille V,
        'endpoints': [
            {
                'Ldc': int,
                'cache_lat': {cache_id: latency, ...}
            }, ...
        ],
        'requests': [
            (video_id, endpoint_id, n_requests), ...
        ]
    }
    """
    path = Path(path)
    print(f"[INFO] Lecture du dataset : {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Dataset introuvable : {path}")

    with path.open("r", encoding="utf-8") as f:
        # Première ligne
        first = f.readline().strip().split()
        if len(first) != 5:
            raise ValueError("Première ligne du fichier invalide (attendu 5 entiers).")
        V, E, R, C, X = map(int, first)

        # Tailles des vidéos
        video_sizes = list(map(int, f.readline().strip().split()))
        if len(video_sizes) != V:
            raise ValueError("Nombre de tailles de vidéos différent de V.")

        endpoints = []
        for e in range(E):
            line = f.readline().strip().split()
            if len(line) != 2:
                raise ValueError(f"Erreur dans la description de l'endpoint {e}.")
            Ldc = int(line[0])
            K = int(line[1])
            cache_lat = {}
            for _ in range(K):
                c_id, Lc = map(int, f.readline().strip().split())
                cache_lat[c_id] = Lc
            endpoints.append({
                "Ldc": Ldc,
                "cache_lat": cache_lat
            })

        requests = []
        for _ in range(R):
            v_id, e_id, nreq = map(int, f.readline().strip().split())
            requests.append((v_id, e_id, nreq))

    print(f"[INFO] Dataset chargé : V={V}, E={E}, R={R}, C={C}, X={X}")
    return {
        "V": V, "E": E, "R": R, "C": C, "X": X,
        "video_sizes": video_sizes,
        "endpoints": endpoints,
        "requests": requests,
    }


def preprocess_requests(data):
    """
    Agrège les requêtes par paire (endpoint, video) pour faciliter le modèle.

    R_e_v[e][v] = total des requêtes de la vidéo v depuis l'endpoint e.
    """
    print("[INFO] Pré-agrégation des requêtes (endpoint, video)...")
    R_e_v = defaultdict(lambda: defaultdict(int))
    for v_id, e_id, nreq in data["requests"]:
        R_e_v[e_id][v_id] += nreq
    return R_e_v


# =========================
#  Modèle Gurobi
# =========================

def build_model(data):
    """
    Construit le modèle Gurobi à partir des données.

    Variables :
        x[v,c] = 1 si vidéo v stockée sur cache c
        y[e,v,c] = 1 si les requêtes (e,v) sont servies par cache c

    Contraintes :
        - capacité des caches : sum_v size[v]*x[v,c] <= X
        - cohérence y <= x
        - pour chaque (e,v) : sum_c y[e,v,c] <= 1

    Objectif :
        Maximise le gain total de latence = sum_{e,v,c} (Ldc[e] - L[e][c]) * n_req[e,v] * y[e,v,c]

    Renvoie (model, x, y, R_e_v).
    """
    V = data["V"]
    E = data["E"]
    C = data["C"]
    X = data["X"]
    video_sizes = data["video_sizes"]
    endpoints = data["endpoints"]

    R_e_v = preprocess_requests(data)

    print("[INFO] Construction du modèle Gurobi...")
    m = gp.Model("hashcode_videos")

    # Paramètres Gurobi
    m.Params.MIPGap = 5e-3  # écart d'optimalité max
    # m.Params.TimeLimit = 300  # optionnel : limite de temps en secondes

    # 1) Variables x[v,c] : vidéo v dans cache c ?
    print("[INFO] Création des variables x[v,c]...")
    x = {}
    for v in range(V):
        if video_sizes[v] > X:
            # vidéo plus grande que la capacité d'un cache : impossible à stocker
            continue
        for c in range(C):
            x[v, c] = m.addVar(
                vtype=GRB.BINARY,
                name=f"x_{v}_{c}"
            )
    m.update()

    # 2) Variables y[e,v,c] : requêtes (e,v) servies par cache c ?
    print("[INFO] Création des variables y[e,v,c]...")
    y = {}
    for e in range(E):
        endpoint = endpoints[e]
        Ldc = endpoint["Ldc"]
        cache_lat = endpoint["cache_lat"]
        for v, nreq in R_e_v[e].items():
            if video_sizes[v] > X:
                continue
            for c, Lc in cache_lat.items():
                # créer y seulement si le cache apporte un gain de latence
                if Ldc > Lc:
                    y[e, v, c] = m.addVar(
                        vtype=GRB.BINARY,
                        name=f"y_{e}_{v}_{c}"
                    )
    m.update()

    # 3) Contraintes de capacité des caches
    print("[INFO] Ajout des contraintes de capacité...")
    for c in range(C):
        expr = gp.LinExpr()
        for v in range(V):
            if (v, c) in x:
                expr += video_sizes[v] * x[v, c]
        if expr.size() > 0:
            m.addConstr(expr <= X, name=f"cap_{c}")

    # 4) Lien y <= x
    print("[INFO] Ajout des contraintes de liaison y <= x...")
    for (e, v, c), y_var in y.items():
        if (v, c) in x:
            m.addConstr(y_var <= x[v, c], name=f"link_{e}_{v}_{c}")
        else:
            # sécurité : si jamais pas de x[v,c], forcer y à 0
            m.addConstr(y_var <= 0, name=f"nolink_{e}_{v}_{c}")

    # 5) Au plus un cache par (e,v)
    print("[INFO] Ajout des contraintes 'au plus un cache par (e,v)'...")
    for e in range(E):
        cache_lat = endpoints[e]["cache_lat"]
        for v in R_e_v[e].keys():
            expr = gp.LinExpr()
            for c in cache_lat.keys():
                if (e, v, c) in y:
                    expr += y[e, v, c]
            if expr.size() > 0:
                m.addConstr(expr <= 1, name=f"onecache_{e}_{v}")

    # 6) Objectif : maximiser le temps total économisé
    print("[INFO] Construction de la fonction objectif...")
    obj = gp.LinExpr()
    for (e, v, c), y_var in y.items():
        endpoint = endpoints[e]
        Ldc = endpoint["Ldc"]
        Lc = endpoint["cache_lat"][c]
        gain_ms = Ldc - Lc  # gain de latence par requête
        nreq = R_e_v[e][v]  # nombre de requêtes pour (e,v)
        coeff = gain_ms * nreq
        obj += coeff * y_var

    m.setObjective(obj, GRB.MAXIMIZE)

    print("[INFO] Modèle construit.")
    return m, x, y, R_e_v


# =========================
#  Écriture des fichiers de sortie
# =========================

def write_mps(model, path="videos.mps"):
    """
    Écrit le modèle Gurobi au format MPS dans le répertoire courant.
    """
    print(f"[INFO] Écriture du modèle au format MPS dans {path}...")
    model.write(path)


def write_solution_output(data, x, out_path="videos.out"):
    """
    Extrait les x[v,c] = 1 et écrit le fichier videos.out au format Hash Code.

    Format attendu :
        L
        c_0 v_0 v_1 v_2 ...
        c_1 v_0 v_3 ...
        ...

    Où L est le nombre de caches ayant au moins une vidéo stockée.
    """
    print(f"[INFO] Écriture de la solution dans {out_path}...")
    C = data["C"]

    # Récupérer les vidéos stockées par cache
    cache_videos = {c: [] for c in range(C)}
    for (v, c), var in x.items():
        if var.X > 0.5:
            cache_videos[c].append(v)

    # Garder seulement les caches non vides
    non_empty = {c: vids for c, vids in cache_videos.items() if len(vids) > 0}

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(str(len(non_empty)) + "\n")
        for c, vids in non_empty.items():
            vids_sorted = sorted(vids)
            line = " ".join(str(v) for v in vids_sorted)
            f.write(f"{c} {line}\n")

    print("[INFO] Fichier videos.out généré.")


# =========================
#  Main
# =========================

def main():
    if len(sys.argv) != 2:
        print("Usage: python videos.py path/to/dataset.in")
        sys.exit(1)

    in_path = sys.argv[1]

    # 1) Lecture des données
    data = parse_input(in_path)

    # 2) Construction du modèle
    model, x, y, R_e_v = build_model(data)

    # 3) Écriture du modèle en .mps dans le répertoire courant
    write_mps(model, "videos.mps")

    # 4) Résolution
    print("[INFO] Lancement de l'optimisation...")
    model.optimize()
    print("[INFO] Optimisation terminée.")

    if model.Status == GRB.OPTIMAL:
        print(f"[INFO] Solution optimale trouvée. Obj = {model.ObjVal}")
    elif model.Status == GRB.SUBOPTIMAL:
        print(f"[WARN] Solution sous-optimale trouvée. Obj = {model.ObjVal}")
    else:
        print(f"[WARN] Statut du modèle : {model.Status}")

    # 5) Écriture de la solution (même si SUBOPTIMAL tant que le gap respecte MIPGap)
    write_solution_output(data, x, "videos.out")



if __name__ == "__main__":
    main()