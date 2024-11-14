import numpy as np
import matplotlib.pyplot as plt
import math


class VectorCache:
    def __init__(self, desired_directions=2, d_strength=0, d_neighbors=1, e_strength=0):
        self.d_strength = d_strength
        self.e_strength = e_strength
        self.d_neighbors = d_neighbors

        if 2 <= desired_directions <= 5:
            self.vectors = self.up_down_vecs()
            self.num_vec = len(self.vectors)
        elif 6 <= desired_directions <= 11:
            self.vectors = self.six_vecs()
            self.num_vec = len(self.vectors)
        else:
            # Mapping of subdivisions to number of vectors
            subdivisions_mapping = {0: 12, 1: 42, 2: 162, 3: 642, 4: 2562}
            # List of allowed directions and corresponding subdivisions
            allowed_directions = [(directions, subd)
                                  for subd, directions in subdivisions_mapping.items()]
            # Filter options less than or equal to desired_directions
            possible_options = [
                option for option in allowed_directions if option[0] <= desired_directions]
            # grab the last one
            self.num_vec, subdivisions = possible_options[-1]
            self.vectors = self.generate_vectors(subdivisions=subdivisions)
        print(f"Vectors may point in {self.num_vec} directions")

        neighbors = max(d_neighbors, 1)
        grid_size = 2 * neighbors + 1

        self.effective_field = np.zeros(
            (self.num_vec, grid_size, grid_size, grid_size, 3), dtype=np.float64)
        for i in range(self.num_vec):
            if d_strength != 0:
                self.effective_field[i] += d_strength * self.dipole_mag_field(
                    self.vectors[i], grid_size=grid_size)
            if e_strength != 0:
                self.effective_field[i] += e_strength * self.exchange_field(
                    self.vectors[i], grid_size=grid_size)
        # print(f"effective field for {self.vectors[0]} spin is: \n{self.effective_field[0]}")

    # Generate MAg field contributions
    def exchange_field(self, s, grid_size=3):
        exchange_field = np.zeros(
            (grid_size, grid_size, grid_size, 3), dtype=np.float64)

        # Calculate the center index of the grid
        center = grid_size // 2

        # Define the six adjacent positions to the center
        adjacent_positions = [
            (center - 1, center, center),  # Left
            (center + 1, center, center),  # Right
            (center, center - 1, center),  # Back
            (center, center + 1, center),  # Front
            (center, center, center - 1),  # Down
            (center, center, center + 1),  # Up
        ]

        # Set the magnetic field at the adjacent positions
        for pos in adjacent_positions:
            x, y, z = pos
            exchange_field[x, y, z] = s

        return exchange_field

    def dipole_mag_field(self, s, grid_size=5):
        half_grid = grid_size // 2

        # Initialize a grid_size x grid_size x grid_size x 3 array to store the magnetic field vectors
        B_vectors = np.zeros(
            (grid_size, grid_size, grid_size, 3), dtype=np.float64)

        for x in range(grid_size):
            for y in range(grid_size):
                for z in range(grid_size):
                    # Position vector r relative to the center
                    # dimensions of lattice const
                    r = np.array([x - half_grid, y - half_grid, z - half_grid])
                    r_magnitude = np.linalg.norm(r)

                    if r_magnitude == 0:
                        continue  # Skip the center point to avoid division by zero

                    # Calculate the magnetic field vector B at this point
                    r_unit = r / r_magnitude
                    dot_product = np.dot(s, r_unit)  # s is in units of mu_s
                    B_vector = (3 * dot_product * r_unit - s) / \
                        r_magnitude**3  # dimensionless

                    B_vectors[x, y, z] = B_vector

        return B_vectors

    def up_down_vecs(self):
        return np.array([[0, 0, 1], [0, 0, -1]], dtype=np.float64)

    def six_vecs(self):
        return np.array([[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]], dtype=np.float64)

    def generate_vectors(self, subdivisions=1):
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

                new_faces.extend(
                    [[v1, a, c], [v2, b, a], [v3, c, b], [a, b, c]])

            faces = new_faces

        # Normalize again once subdivided
        normalize_vertices(vertices)
        # size is constant so cast to a numpy array
        return np.array(vertices, dtype=np.float64)

    def visualize_effective_field(self, s):
        grid_size = len(self.effective_field[0])
        field = np.zeros_like(self.effective_field[0])

        field += self.d_strength * self.dipole_mag_field(
            s, grid_size=grid_size)

        field += self.e_strength * self.exchange_field(
            s, grid_size=grid_size)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        Y, X, Z = np.meshgrid(np.arange(grid_size), np.arange(
            grid_size), np.arange(grid_size))

        u = field[..., 0]  # X component of B
        v = field[..., 1]  # Y component of B
        w = field[..., 2]  # Z component of B

        ax.quiver(X, Y, Z, u, v, w, length=0.2)

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        min_limit = min(np.min(X), np.min(Y), np.min(Z))
        max_limit = max(np.max(X), np.max(Y), np.max(Z))

        ax.set_xlim(min_limit, max_limit)
        ax.set_ylim(min_limit, max_limit)
        ax.set_zlim(min_limit, max_limit)

        plt.show()


if __name__ == "__main__":
    for subdivisions in range(5):
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
