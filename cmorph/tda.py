from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple

def lower_star_persistence_0d(path: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Persistance 0D exacte (composantes connexes) pour une courbe 1D discrète.
    On utilise une filtration 'lower-star': on active les sommets par ordre croissant de valeur,
    et on connecte chaque sommet à ses voisins immédiats déjà actifs.
    Retourne:
      births: naissance (valeur) de chaque composante (min local),
      deaths: mort (valeur) de la composante (lors fusion avec une composante plus 'ancienne').
    Convention: la dernière composante (globale) a death = +inf (on l'ignore pour les métriques).
    """
    x = np.asarray(path, dtype=float)
    n = len(x)
    order = np.argsort(x)  # indices triés par hauteur
    parent = np.arange(n)  # union-find
    active = np.zeros(n, dtype=bool)
    birth_val = np.full(n, np.nan)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    births: List[float] = []
    deaths: List[float] = []

    for idx in order:
        active[idx] = True
        parent[idx] = idx
        birth_val[idx] = x[idx]  # nouvelle composante naît ici

        # check voisins (1D)
        for nb in (idx-1, idx+1):
            if 0 <= nb < n and active[nb]:
                ra, rb = find(idx), find(nb)
                if ra == rb:
                    continue
                # on fusionne: la composante avec le PLUS PETIT birth survit (elder rule)
                if birth_val[ra] <= birth_val[rb]:
                    # rb meurt à la valeur courante x[idx]
                    parent[rb] = ra
                    births.append(birth_val[rb])
                    deaths.append(x[idx])
                else:
                    parent[ra] = rb
                    births.append(birth_val[ra])
                    deaths.append(x[idx])

    births = np.asarray(births, dtype=float)
    deaths = np.asarray(deaths, dtype=float)
    return births, deaths

def tda_features_lower_star(path: np.ndarray) -> Dict[str, float]:
    """
    Calcule quelques features stables à partir des paires (birth, death) 0D
    (on ignore la dernière composante 'immortelle').
    - pers_max : max(death-birth)
    - pers_sum : somme des persistences
    - pers_entropy : entropie normalisée des persistences
    - n_pairs : nombre de paires finies
    """
    births, deaths = lower_star_persistence_0d(path)
    if len(births) == 0:
        return {"pers_max": 0.0, "pers_sum": 0.0, "pers_entropy": 0.0, "n_pairs": 0.0}
    pers = np.maximum(0.0, deaths - births)
    pers_sum = float(np.sum(pers))
    pers_max = float(np.max(pers))
    p = pers / pers_sum if pers_sum > 0 else np.ones_like(pers) / len(pers)
    eps = 1e-12
    ent = float(-np.sum(p * np.log(p + eps)))
    return {
        "pers_max": pers_max,
        "pers_sum": pers_sum,
        "pers_entropy": ent,
        "n_pairs": float(len(pers)),
    }
