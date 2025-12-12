
Fake News Blocker Placement Experiments
# =====================================

import math
import random
import itertools
from dataclasses import dataclass
from typing import List, Set, Tuple, Optional

import matplotlib.pyplot as plt


# =========================
# Graph Representation
# =========================

@dataclass
class Graph:
    n: int
    edges: Set[Tuple[int, int]]

    def __init__(self, n: int):
        self.n = n
        self.edges = set()

    def add_edge(self, u: int, v: int) -> None:
        if u == v:
            return
        if v < u:
            u, v = v, u
        self.edges.add((u, v))

    def neighbors(self, u: int) -> Set[int]:
        nbrs: Set[int] = set()
        for (a, b) in self.edges:
            if a == u:
                nbrs.add(b)
            elif b == u:
                nbrs.add(a)
        return nbrs


# =========================
# Greedy 2-Approx Vertex Cover
# =========================

def greedy_vertex_cover(g: Graph) -> Set[int]:
    """
    Simple 2-approximation:
      While there is an uncovered edge (u,v),
      add both u and v to the cover and remove all incident edges.
    """
    cover: Set[int] = set()
    uncovered_edges = set(g.edges)

    while uncovered_edges:
        (u, v) = next(iter(uncovered_edges))
        cover.add(u)
        cover.add(v)

        to_remove = []
        for (a, b) in uncovered_edges:
            if a in (u, v) or b in (u, v):
                to_remove.append((a, b))
        for e in to_remove:
            uncovered_edges.remove(e)

    return cover


# =========================
# Exact Vertex Cover (Brute Force)
# =========================

def is_vertex_cover(g: Graph, cover: Set[int]) -> bool:
    for (u, v) in g.edges:
        if u not in cover and v not in cover:
            return False
    return True


def brute_force_min_vertex_cover(g: Graph, max_n: int = 20) -> Optional[Set[int]]:
    """
    Exponential-time exact solver; use only for small n.
    """
    n = g.n
    if n > max_n:
        return None

    vertices = list(range(n))
    best_cover: Optional[Set[int]] = None

    for k in range(n + 1):
        for combo in itertools.combinations(vertices, k):
            cover = set(combo)
            if is_vertex_cover(g, cover):
                return cover  # first found is minimum for this k
    return best_cover


# =========================
# Random Graph Generator G(n, p)
# =========================

def random_gnp_graph(n: int, p: float) -> Graph:
    g = Graph(n)
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < p:
                g.add_edge(u, v)
    return g


# =========================
# Experiment 1: Runtime vs n
# =========================

def experiment_runtime_vs_n():
    ns = [200, 400, 800, 1600]
    p = 0.02
    trials = 5

    avg_runtimes_ms: List[float] = []

    for n in ns:
        import time
        total = 0.0
        print(f"[Runtime] n={n}")
        for _ in range(trials):
            g = random_gnp_graph(n, p)
            start = time.perf_counter()
            _ = greedy_vertex_cover(g)
            end = time.perf_counter()
            total += (end - start)
        avg = (total / trials) * 1000.0  # milliseconds
        avg_runtimes_ms.append(avg)
        print(f"  avg runtime: {avg:.4f} ms")

    plt.figure(figsize=(6, 4))
    plt.plot(ns, avg_runtimes_ms, marker="o")
    plt.title("Greedy Fake News Blocker Runtime vs n (G(n, p))")
    plt.xlabel("Number of vertices (n)")
    plt.ylabel("Average runtime (ms)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("runtime_vs_n.png")
    plt.close()

    return ns, avg_runtimes_ms


# =========================
# Experiment 2: Approximation Ratio vs n (small graphs)
# =========================

def experiment_approx_ratio_small_graphs():
    ns = [8, 10, 12, 14, 16]
    p = 0.3
    trials = 10

    avg_ratios: List[float] = []

    for n in ns:
        ratios: List[float] = []
        print(f"[ApproxRatio] n={n}")
        for _ in range(trials):
            g = random_gnp_graph(n, p)
            greedy_cover = greedy_vertex_cover(g)
            optimal_cover = brute_force_min_vertex_cover(g, max_n=20)
            if not optimal_cover:
                continue
            ratio = len(greedy_cover) / len(optimal_cover) if len(optimal_cover) > 0 else float("inf")
            ratios.append(ratio)
        avg_ratio = sum(ratios) / len(ratios) if ratios else float("inf")
        avg_ratios.append(avg_ratio)
        print(f"  avg ratio |Greedy|/|Optimal| = {avg_ratio:.4f}")

    plt.figure(figsize=(6, 4))
    plt.plot(ns, avg_ratios, marker="o")
    plt.title("Greedy Blocker Placement Approximation Ratio vs n")
    plt.xlabel("Number of vertices (n)")
    plt.ylabel("Average |Greedy| / |Optimal|")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("approx_ratio_small_graphs.png")
    plt.close()

    return ns, avg_ratios


# =========================
# Visualization: Blocker Placement on One Graph
# =========================

def visualize_blocker_placement():
    n = 200
    p = 0.05
    g = random_gnp_graph(n, p)
    cover = greedy_vertex_cover(g)

    # Circular layout
    angles = [2.0 * math.pi * i / n for i in range(n)]
    xs = [math.cos(a) for a in angles]
    ys = [math.sin(a) for a in angles]

    plt.figure(figsize=(8, 8))

    # Draw edges in light gray
    for (u, v) in g.edges:
        x_vals = [xs[u], xs[v]]
        y_vals = [ys[u], ys[v]]
        plt.plot(x_vals, y_vals, linewidth=0.3, alpha=0.3)

    # Draw nodes: blockers red, regular blue
    bx = [xs[i] for i in range(n) if i in cover]
    by = [ys[i] for i in range(n) if i in cover]
    rx = [xs[i] for i in range(n) if i not in cover]
    ry = [ys[i] for i in range(n) if i not in cover]

    plt.scatter(rx, ry, s=30, label="Regular account")
    plt.scatter(bx, by, s=30, label="Blocker account")

    plt.axis("off")
    plt.title("Fake News Blocker Placement (Vertex Cover)")

    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("graph_output.png")
    plt.close()

    return g, cover


# =========================
# Main
# =========================

def main():
    experiment_runtime_vs_n()
    experiment_approx_ratio_small_graphs()
    visualize_blocker_placement()
    print("Generated: runtime_vs_n.png, approx_ratio_small_graphs.png, graph_output.png")


if __name__ == "__main__":
    main()
