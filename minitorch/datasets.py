import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List[Tuple[float, float]]:
    """Generates a list of N random points in the unit square [0, 1] x [0, 1].

    Args:
    ----
        N: The number of points to generate.

    Returns:
    -------
        A list of N tuples, where each tuple contains two float values between 0 and 1.

    """
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    """A graph is a dataset with N points, X, and a list of labels, y.

    Args:
    ----
        N: The number of points in the graph.
        X: A list of tuples, where each tuple contains two float values between 0 and 1.
        y: A list of integers, where each integer is either 0 or 1.

    """

    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """Generates a simple graph with N points.

    Args:
    ----
        N: The number of points in the graph.

    Returns:
    -------
        A Graph object with N points and labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """Generates a diagonal graph with N points.

    Args:
    ----
        N: The number of points in the graph.

    Returns:
    -------
        A Graph object with N points and labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """Generates a split graph with N points.

    Args:
    ----
        N: The number of points in the graph.

    Returns:
    -------
        A Graph object with N points and labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """Generates an XOR graph with N points.

    Args:
    ----
        N: The number of points in the graph.

    Returns:
    -------
        A Graph object with N points and labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 and x_2 > 0.5 or x_1 > 0.5 and x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """Generates a circle graph with N points.

    Args:
    ----
        N: The number of points in the graph.

    Returns:
    -------
        A Graph object with N points and labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = x_1 - 0.5, x_2 - 0.5
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """Generates a spiral graph with N points.

    Args:
    ----
        N: The number of points in the graph.

    Returns:
    -------
        A Graph object with N points and labels.

    """

    def x(t: float) -> float:
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
