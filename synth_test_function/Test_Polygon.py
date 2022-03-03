from shapely.geometry import Polygon
import matplotlib.pyplot as plt


def test_contains():
    poly_a = Polygon([(0, 0), (0, 1), (1, 0)])
    poly_b = Polygon([(0.5, 0.8), (0.8, 0.8), (0.8, 0.5)])
    x, y = poly_a.exterior.xy
    print(list(x))
    plt.plot(x, y)
    x, y = poly_b.exterior.xy
    plt.plot(x, y)
    plt.show()
    print("A có nằm trong B hay không ? ", poly_a.contains(poly_b))


def test_overlap():
    p1 = Polygon([(0, 0), (1, 1), (1, 0)])
    p2 = Polygon([(0, 1), (0.8, 0.9), (0.2, 0.4)])
    x, y = p1.exterior.xy
    plt.plot(x, y)
    x, y = p2.exterior.xy
    plt.plot(x, y)
    plt.show()
    print("A có overlap B hay không ? ", p1.intersects(p2))


if __name__ == "__main__":
    pass
