import numpy as np
import math


class VectorCache:
    def __init__(self, subdivisions=1, neighbors=2):
        self.vectors = self.generate_vectors(subdivisions)
        self.num_vec = len(self.vectors)
        grid_size = 2 * neighbors + 1
        self.dipole_contributions = np.zeros(
            (self.num_vec, grid_size, grid_size, grid_size, 3), dtype=np.float64)
        for i in range(self.num_vec):
            self.dipole_contributions[i] += self.calc_simp_mag_field( 
                self.vectors[i], mu=1, grid_size=grid_size)

    #Generate MAg field contributions
    def calc_simp_mag_field(self, m, mu=1, grid_size=3):
        half_grid = grid_size // 2

        # Initialize a grid_size x grid_size x grid_size x 3 array to store the magnetic field vectors
        B_vectors = np.full((grid_size, grid_size, grid_size, 3), m*mu)
        B_vectors[half_grid, half_grid, half_grid] = 0

        return B_vectors


    def calculate_magnetic_field(self, m, mu=1, grid_size=5):
        half_grid = grid_size // 2

        # Initialize a grid_size x grid_size x grid_size x 3 array to store the magnetic field vectors
        B_vectors = np.zeros((grid_size, grid_size, grid_size, 3))

        for x in range(grid_size):
            for y in range(grid_size):
                for z in range(grid_size):
                    # Position vector r relative to the center
                    r = np.array([x - half_grid, y - half_grid, z - half_grid])
                    r_magnitude = np.linalg.norm(r)

                    if r_magnitude == 0:
                        continue  # Skip the center point to avoid division by zero

                    # Calculate the magnetic field vector B at this point
                    r_unit = r / r_magnitude
                    dot_product = np.dot(m, r_unit)
                    B_vector = (mu / (4 * np.pi)) * (
                        (3 * dot_product * r_unit - m) / r_magnitude**3
                    )

                    B_vectors[x, y, z] = B_vector

        return B_vectors

    def generate_vectors(self, subdivisions = 1):
        def midpoint(v1, v2):
            return [(v1[i] + v2[i]) / 2 for i in range(len(v1))]

        def normalize_vertices(vertices):
            for i in range(len(vertices)):
                norm = math.sqrt(sum(x**2 for x in vertices[i]))
                vertices[i] = [x / norm for x in vertices[i]]
            return vertices
        
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
        faces = [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ]

        normalize_vertices(vertices)

        # Subdivide the icosahedron
        for _ in range(subdivisions):
            new_faces = []  # each face is divided into four, this is the new list
            # Dictionary where value is the index in vertices and the key is 
            # a tuple of the vertices used to make midpoint
            midpoint_cache = {}

            for face in faces:
                v1, v2, v3 = face

                # the three midpoints as defined by their vertices
                # they are sorted so a different order is not unique
                a = tuple(sorted([v1, v2]))
                b = tuple(sorted([v2, v3]))
                c = tuple(sorted([v3, v1]))

                # Create midpoints
                if a not in midpoint_cache:
                    midpoint_cache[a] = len(vertices)
                    vertices.append(midpoint(vertices[v1], vertices[v2]))
                if b not in midpoint_cache:
                    midpoint_cache[b] = len(vertices)
                    vertices.append(midpoint(vertices[v2], vertices[v3]))
                if c not in midpoint_cache:
                    midpoint_cache[c] = len(vertices)
                    vertices.append(midpoint(vertices[v3], vertices[v1]))

                # Extract the index from midpoint cache
                a = midpoint_cache[a]
                b = midpoint_cache[b]
                c = midpoint_cache[c]

                new_faces.extend([[v1, a, c], [v2, b, a], [v3, c, b], [a, b, c]])

            faces = new_faces
        
        # Normalize again once subdivided
        normalize_vertices(vertices)
        # size is constant so cast to a numpy array
        return np.array(vertices, dtype=np.float64)


if __name__ == "__main__":
    subdivisions = 1
    vec = VectorCache(subdivisions, neighbors=1)
    print(
        f"A class 1 Geodsic icosahedron with {subdivisions} subdivisions has {len(vec.vectors)} vertices.")
    print(
        f"Dimensions of the dipole_contribution data is {vec.dipole_contributions.shape}")


def plot_geodesic_dome(vertices):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from scipy.spatial import ConvexHull

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Create the convex hull to get the correct faces
        hull = ConvexHull(vertices)
        for simplex in hull.simplices:
            ax.add_collection3d(Poly3DCollection(
                [vertices[simplex]], facecolors='c', linewidths=1, edgecolors='r', alpha=.25))

        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='b')

        # Equal scaling
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Geodesic Dome (Class I)')