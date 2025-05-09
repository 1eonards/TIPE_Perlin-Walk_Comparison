import numpy as np
from noise import pnoise2

from numpy.ma.extras import average


def generate_random_walk_terrain(width, height, step_size=0.05, seed=0):
    """
    Génère un terrain en utilisant une marche aléatoire simplifiée
    Parameters:
        width, height : dimensions du terrain
        step_size : amplitude du pas aléatoire
        seed : graine pour la reproductibilité
    Returns:
        terrain : tableau 2D représentant l'élévation du terrain
    """
    start_time = time.time()
    np.random.seed(seed)
    terrain = np.zeros((height, width))

    # On commence au centre avec une valeur moyenne
    terrain[0][0] = 0.5

    # Génération ligne par ligne
    for y in range(height):
        for x in range(width):
            # Pour le premier point de chaque ligne/colonne, on se base sur les voisins déjà calculés
            if x == 0 and y > 0:
                terrain[y][x] = terrain[y - 1][x] + (np.random.random() - 0.5) * 2 * step_size
            elif x > 0:
                terrain[y][x] = terrain[y][x - 1] + (np.random.random() - 0.5) * 2 * step_size

            # On s'assure que les valeurs restent entre 0 et 1
            terrain[y][x] = max(0, min(1, terrain[y][x]))
    end_time = time.time()
    return 150 * terrain, end_time - start_time


def generate_perlin_terrain(width, height, scale=100.0, octaves=6,
                            persistence=0.5, lacunarity=2.0, seed=0):
    """
    Génère un terrain en utilisant le bruit de Perlin
    Parameters:
        width, height : dimensions du terrain
        scale : échelle du bruit (valeurs plus élevées = relief plus lisse)
        octaves : nombre de couches de bruit superposées
        persistence : influence des octaves supérieures
        lacunarity : fréquence des octaves supérieures
        seed : graine pour la reproductibilité
    Returns:
        terrain : tableau 2D représentant l'élévation du terrain
    """
    start_time = time.time()
    terrain = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            terrain[y][x] = pnoise2(x / scale, y / scale,
                                    octaves=octaves,
                                    persistence=persistence,
                                    lacunarity=lacunarity,
                                    repeatx=width,
                                    repeaty=height,
                                    base=seed)

    # Normalisation entre 0 et 1
    terrain = (terrain - np.min(terrain)) / (np.max(terrain) - np.min(terrain))
    end_time = time.time()
    return 150 * terrain, end_time - start_time


def measure_coherence(terrain):
    """
    Mesure la cohérence spatiale en calculant la différence moyenne
    d'élévation entre pixels adjacents
    Une valeur plus faible indique un terrain plus cohérent
    """
    height, width = terrain.shape
    total_diff = 0
    count = 0

    for y in range(height):
        for x in range(width):
            # Pour chaque voisin
            for ny, nx in [(y + 1, x), (y, x + 1)]:  # Seulement droite et bas pour éviter les doublons
                if 0 <= ny < height and 0 <= nx < width:
                    diff = abs(terrain[y][x] - terrain[ny][nx])
                    total_diff += diff
                    count += 1

    return total_diff / count if count > 0 else 0

import time

def measure_performance(generator_func, width, height, **kwargs):
    """
    Mesure le temps d'exécution d'un générateur de terrain
    """
    start_time = time.time()
    generator_func(width, height, **kwargs)
    end_time = time.time()
    return end_time - start_time


def run_experiment():
    results = {
        'perlin': {
            'coherence': [],
            'performance': []
        },
        'random_walk': {
            'coherence': [],
            'performance': []
        }
    }

    width, height = 512, 512
    seeds = range(10)  # 10 terrains différents

    for seed in seeds:
        print(seed)
        # Génération de terrains
        perlin_terrain = generate_perlin_terrain(width, height, seed=seed)
        random_walk_terrain = generate_random_walk_terrain(width, height, seed=seed)

        # Mesures
        for terrain_type, (terrain, performance) in [('perlin', perlin_terrain),
                                      ('random_walk', random_walk_terrain)]:
            results[terrain_type]['coherence'].append(measure_coherence(terrain))
            results[terrain_type]['performance'].append(performance)
    # Moyenne des mesures obtenues
    results['perlin']['coherence'] = average(results['perlin']['coherence'])
    results['perlin']['performance'] = average(results['perlin']['performance'])
    results['random_walk']['coherence'] = average(results['random_walk']['coherence'])
    results['random_walk']['performance'] = average(results['random_walk']['performance'])
    return results

if __name__ == "__main__":
    experiment = run_experiment()
    print(f"La cohérence moyenne pour la marche aléatoire est de {experiment['random_walk']['coherence']:.2f}")
    print(f"Le temps d'exécution moyen pour la marche aléatoire est de {experiment['random_walk']['performance']:.2f}s")
    print(f"La cohérence moyenne pour le bruit de Perlin est de {experiment['perlin']['coherence']:.2f}")
    print(f"Le temps d'exécution moyen pour le bruit de Perlin est de {experiment['perlin']['performance']:.2f}s")
