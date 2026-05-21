import sys
import os
import matplotlib.pyplot as plt

def parse_points(text):
    points = []
    for line in text.strip().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            points.append((float(parts[0]), float(parts[1])))
    return points

if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    points = parse_points(text)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    plt.scatter(xs, ys, s=1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Points")
    plt.savefig(os.path.dirname(__file__) + "/graph.png")
    print("Saved graph.png")
