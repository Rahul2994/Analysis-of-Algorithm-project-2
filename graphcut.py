Full Graph-Cut Segmentation + Runtime Experiment Code
-----------------------------------------------------
import time
import random
import numpy as np
import matplotlib.pyplot as plt


# ================================================================
# 1. Dinic's Algorithm for Max Flow
# ================================================================

class Edge:
    def __init__(self, to, capacity, rev):
        self.to = to
        self.capacity = capacity
        self.rev = rev  # reverse edge index


class Dinic:
    def __init__(self, n):
        self.n = n
        self.adj = [[] for _ in range(n)]

    def add_edge(self, fr, to, cap):
        fwd = Edge(to, cap, len(self.adj[to]))
        rev = Edge(fr, 0.0, len(self.adj[fr]))
        self.adj[fr].append(fwd)
        self.adj[to].append(rev)

    def bfs(self, s, t, level):
        for i in range(self.n):
            level[i] = -1
        queue = [s]
        level[s] = 0

        for v in queue:
            for e in self.adj[v]:
                if e.capacity > 1e-12 and level[e.to] < 0:
                    level[e.to] = level[v] + 1
                    queue.append(e.to)

        return level[t] != -1

    def send_flow(self, v, t, f, level, work):
        if v == t:
            return f

        for i in range(work[v], len(self.adj[v])):
            e = self.adj[v][i]

            if e.capacity > 1e-12 and level[e.to] == level[v] + 1:
                pushed = self.send_flow(e.to, t, min(f, e.capacity), level, work)
                if pushed > 0:
                    e.capacity -= pushed
                    self.adj[e.to][e.rev].capacity += pushed
                    return pushed

            work[v] += 1

        return 0

    def max_flow(self, s, t):
        flow = 0
        level = [-1] * self.n

        while self.bfs(s, t, level):
            work = [0] * self.n
            while True:
                pushed = self.send_flow(s, t, float("inf"), level, work)
                if pushed <= 0:
                    break
                flow += pushed

        return flow


# ================================================================
# 2. Graph-Cut Segmentation for a Grayscale Image
# ================================================================

def segment_graph_cut(img, seeds_fg, seeds_bg, lam=50.0):
    """
    img         : H×W grayscale image (0–255)
    seeds_fg    : list of (r,c) seed pixels for foreground
    seeds_bg    : list of (r,c) seed pixels for background
    """
    H, W = img.shape
    N = H * W
    S = N     # source
    T = N + 1 # sink
    dinic = Dinic(N + 2)

    def node(r, c):
        return r * W + c

    # Smoothness term — penalize discontinuities
    def nlink_weight(p, q):
        return lam * np.exp(-((p - q) ** 2) / 50.0)

    # ----------------------------------------------------------
    # Build graph
    # ----------------------------------------------------------
    for r in range(H):
        for c in range(W):
            u = node(r, c)
            intensity = img[r, c]

            # Regional terms (t-links)
            dinic.add_edge(S, u, 0.0)
            dinic.add_edge(u, T, 0.0)

            # Smoothness edges (n-links)
            if r + 1 < H:
                w = nlink_weight(intensity, img[r+1, c])
                dinic.add_edge(u, node(r+1, c), w)
                dinic.add_edge(node(r+1, c), u, w)
            if c + 1 < W:
                w = nlink_weight(intensity, img[r, c+1])
                dinic.add_edge(u, node(r, c+1), w)
                dinic.add_edge(node(r, c+1), u, w)

    BIG = 10**9

    # Foreground seeds → connect to SOURCE strongly
    for (r, c) in seeds_fg:
        dinic.add_edge(S, node(r, c), BIG)

    # Background seeds → connect to SINK strongly
    for (r, c) in seeds_bg:
        dinic.add_edge(node(r, c), T, BIG)

    # ----------------------------------------------------------
    # Compute max-flow = min-cut
    # ----------------------------------------------------------
    dinic.max_flow(S, T)

    # ----------------------------------------------------------
    # Recover segmentation via BFS on residual graph from source
    # ----------------------------------------------------------
    visited = [False] * (N + 2)
    queue = [S]
    visited[S] = True

    for v in queue:
        for e in dinic.adj[v]:
            if e.capacity > 1e-12 and not visited[e.to]:
                visited[e.to] = True
                queue.append(e.to)

    mask = np.zeros((H, W), dtype=np.uint8)
    for r in range(H):
        for c in range(W):
            if visited[node(r, c)]:  # reachable from source → FG
                mask[r, c] = 255

    return mask


# ================================================================
# 3. Synthetic Image Generator
# ================================================================

def make_synthetic_image(H, W):
    return np.random.randint(0, 255, size=(H, W), dtype=np.uint8)


# ================================================================
# 4. Runtime Experiments
# ================================================================

def run_experiments():
    sizes = [100*100, 150*150, 200*200, 250*250, 320*320]
    runtimes = []

    for pixels in sizes:
        H = int(np.sqrt(pixels))
        W = H

        print(f"Running segmentation for {H}×{W} ({pixels} pixels) ...")

        img = make_synthetic_image(H, W)

        seeds_fg = [(10, 10)]
        seeds_bg = [(H-10, W-10)]

        trials = 3
        total = 0.0

        for _ in range(trials):
            t0 = time.time()
            _ = segment_graph_cut(img, seeds_fg, seeds_bg)
            t1 = time.time()
            total += (t1 - t0)

        avg_time = total / trials
        runtimes.append(avg_time)
        print(f"  → Avg runtime: {avg_time:.4f} sec")

    return sizes, runtimes


# ================================================================
# 5. Plotting Function
# ================================================================

def plot_results(sizes, runtimes):
    plt.figure(figsize=(8, 6))
    plt.plot(sizes, runtimes, marker='o')
    plt.title("Graph-cut segmentation runtime vs. image size")
    plt.xlabel("Number of pixels (H × W)")
    plt.ylabel("Average runtime (seconds)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("runtime_vs_pixels.png")
    plt.show()


# ================================================================
# 6. Main
# ================================================================

if __name__ == "__main__":
    sizes, runtimes = run_experiments()
    plot_results(sizes, runtimes)

    print("\nExperiment completed.")
    print("Saved plot: runtime_vs_pixels.png")
