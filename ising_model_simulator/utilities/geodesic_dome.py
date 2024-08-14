import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

class GeodesicDome:
    def __init__(self, subdivisions: 0):
        self.faces = []
        vertices = self.generate_icosahedron()
        vertices = self.subdivide(subdivisions, vertices)
        self.vertices = np.array(vertices)

    def generate_icosahedron(self):
        phi = (1 + math.sqrt(5)) / 2  # Golden ratio

        vertices = [
            [-1,  phi,  0],
            [1,  phi,  0],
            [-1, -phi,  0],
            [1, -phi,  0],
            [0, -1,  phi],
            [0,  1,  phi],
            [0, -1, -phi],
            [0,  1, -phi],
            [phi,  0, -1],
            [phi,  0,  1],
            [-phi,  0, -1],
            [-phi,  0,  1]
        ]

        # A face is defined by the index of the three points in the triangle
        self.faces = [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ]

        self.normalize_vertices(vertices)
    
        return vertices

    def midpoint(self, v1, v2):
        return [(v1[i] + v2[i]) / 2 for i in range(len(v1))]

    def subdivide(self, subdivisions, vertices):
        for _ in range(subdivisions):
            new_faces = []  # each face is divided into four, this is the new list
            # Dictionary where value is the index in vertices and the key is 
            # a tuple of the vertices used to make midpoint
            midpoint_cache = {}

            for face in self.faces:
                v1, v2, v3 = face

                # the three midpoints as defined by their vertices
                # they are sorted so a different order is not unique
                a = tuple(sorted([v1, v2]))
                b = tuple(sorted([v2, v3]))
                c = tuple(sorted([v3, v1]))

                # Create midpoints
                if a not in midpoint_cache:
                    midpoint_cache[a] = len(vertices)
                    vertices.append(self.midpoint(vertices[v1], vertices[v2]))
                if b not in midpoint_cache:
                    midpoint_cache[b] = len(vertices)
                    vertices.append(self.midpoint(vertices[v2], vertices[v3]))
                if c not in midpoint_cache:
                    midpoint_cache[c] = len(vertices)
                    vertices.append(self.midpoint(vertices[v3], vertices[v1]))

                # Extract the index from midpoint cache
                a = midpoint_cache[a]
                b = midpoint_cache[b]
                c = midpoint_cache[c]

                new_faces.extend([[v1, a, c], [v2, b, a], [v3, c, b], [a, b, c]])

            self.faces = new_faces

        self.normalize_vertices(vertices)
        return vertices


    def normalize_vertices(self, vertices):
        for i in range(len(vertices)):
            norm = math.sqrt(sum(x**2 for x in vertices[i]))
            vertices[i] = [x / norm for x in vertices[i]]
        return vertices


    def plot_geodesic_dome(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Create the convex hull to get the correct faces
        hull = ConvexHull(self.vertices)
        for simplex in hull.simplices:
            ax.add_collection3d(Poly3DCollection(
                [self.vertices[simplex]], facecolors='c', linewidths=1, edgecolors='r', alpha=.25))

        ax.scatter(self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2], color='b')

        # Equal scaling
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Geodesic Dome (Class I)')

        plt.show()

if __name__ == "__main__":
    subdivisions = 1
    dome = GeodesicDome(subdivisions)
    print(f"A class 1 Geodsic icosahedron with {subdivisions} subdivisions has {len(dome.vertices)} vertices and {len(dome.faces)} faces.")
    dome.plot_geodesic_dome()
