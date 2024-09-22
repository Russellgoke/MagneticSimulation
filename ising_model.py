from utilities.vector_cache import VectorCache
import matplotlib.pyplot as plt

import numpy as np


class IsingModel:
    def __init__(self, size=5, external_field=(0, 0, 0), subdivisions=0, neighbors=4):
        self.size = size
        self.vcache = VectorCache(
            subdivisions=subdivisions, neighbors=neighbors)
        self.lattice = self.init_lattice()
        self.mag_field = np.full((self.size, self.size, self.size, 3),
                                 external_field, dtype=np.float64)
        self.init_mag_field()

    def init_lattice(self):
        lattice = np.random.randint(
            low=0,
            high=self.vcache.num_vec,
            size=(self.size, self.size, self.size),
            dtype=np.uint8  # TODO make work with larger too
        )
        return lattice

    def init_mag_field(self):
        for x in range(self.size):
            for y in range(self.size):
                for z in range(self.size):
                    vec_num = self.lattice[x][y][z]
                    self.stamp_onto_field(
                        (x, y, z), self.vcache.dipole_contributions[vec_num])

    def stamp_onto_field(self, center, stamp):
        radius = stamp.shape[0] // 2  # Assuming the stamp is cubic

        # Create index arrays for x, y, z coordinates with wrapping
        x_indices = (np.arange(center[0] - radius,
                     center[0] + radius + 1) % self.size)
        y_indices = (np.arange(center[1] - radius,
                     center[1] + radius + 1) % self.size)
        z_indices = (np.arange(center[2] - radius,
                     center[2] + radius + 1) % self.size)

        # Use np.ix_ to create a 3D grid of indices for stamping the smaller array
        ix, iy, iz = np.ix_(x_indices, y_indices, z_indices)

        # Stamp the smaller array onto the larger array using advanced indexing
        self.mag_field[ix, iy, iz] += stamp

    def run_simulation(self, iterations):
        for _ in range(iterations):
            self.update_lattice()

    def update_lattice(self):
        # TODO: Implement
        pass

    def save_results(self, filename):
        try:
            with open(filename, 'w') as out_file:
                for plane in self.lattice:
                    for row in plane:
                        out_file.write(' '.join(map(str, row)) + '\n')
                    out_file.write('\n')
        except IOError as e:
            print(f"Unable to open file: {filename} - {e}")

    def visualize_magnetic_field(self):
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        Y, X, Z = np.meshgrid(np.arange(self.size), np.arange(
            self.size), np.arange(self.size))

        u = self.mag_field[..., 0]  # X component of B
        v = self.mag_field[..., 1]  # Y component of B
        w = self.mag_field[..., 2]  # Z component of B

        ax.quiver(X, Y, Z, u, v, w, length=0.05)

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()

    def visualize_lattice(self):
        lattice_vecs = np.zeros_like(self.mag_field)
        for x in range(self.size):
            for y in range(self.size):
                for z in range(self.size):
                    lattice_vecs[x][y][z] = self.vcache.vectors[self.lattice[x][y][z]]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        Y, X, Z = np.meshgrid(np.arange(self.size), np.arange(
            self.size), np.arange(self.size))

        u = lattice_vecs[..., 0]  # X component of B
        v = lattice_vecs[..., 1]  # Y component of B
        w = lattice_vecs[..., 2]  # Z component of B

        ax.quiver(X, Y, Z, u, v, w, length=0.5)

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()
        


def test_stamp():
    # neighbors=1 means that this inital settup has no magnetic field at all
    model = IsingModel(5, external_field=(0, 0, 0),
                       subdivisions=0, neighbors=0)
    # A test vector, it is expanding from center equally
    vectors = np.array([
        [
            [[-0.57735027, -0.57735027, -0.57735027], [-0.70710678, -
                                                       0.70710678,  0.0], [-0.57735027, -0.57735027,  0.57735027]],
            [[-0.70710678,  0.0, -0.70710678], [-1.0, 0.0,  0.0],
                [-0.70710678,  0.0,  0.70710678]],
            [[-0.57735027,  0.57735027, -0.57735027], [-0.70710678,
                                                       0.70710678,  0.0], [-0.57735027,  0.57735027,  0.57735027]]
        ],
        [
            [[0.0, -0.70710678, -0.70710678], [0.0, -1.0,  0.0],
                [0.0, -0.70710678,  0.70710678]],
            [[0.0,  0.0, -1.0], [0.0,  0.0,  0.0], [0.0,  0.0,  1.0]],
            [[0.0,  0.70710678, -0.70710678], [0.0,  1.0,  0.0],
                [0.0,  0.70710678,  0.70710678]]
        ],
        [
            [[0.57735027, -0.57735027, -0.57735027], [0.70710678, -
                                                      0.70710678,  0.0], [0.57735027, -0.57735027,  0.57735027]],
            [[0.70710678,  0.0, -0.70710678], [1.0, 0.0,  0.0],
                [0.70710678,  0.0,  0.70710678]],
            [[0.57735027,  0.57735027, -0.57735027], [0.70710678,
                                                      0.70710678,  0.0], [0.57735027,  0.57735027,  0.57735027]]
        ]
    ], dtype=np.float64)
    model.stamp_onto_field((0, 1, 2), vectors)
    # model.stamp_onto_field((0, 0, 0), vectors)
    model.visualize_magnetic_field()


def test_field_init():
    model = IsingModel(size=5, external_field=(
        0, 0, 0), subdivisions=0, neighbors=0)
    # these vectors form a 4 by 3 in space to try and help visualize, num_vec = 12
    test_vecs = [[1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [1, 0, -1], [0, 1, -1], [-1, 0, -1], [0, -1, -1]]
    model.vcache.dipole_contributions = np.reshape(test_vecs, (12, 1, 1, 1, 3))
    # reset and re init mag field with fake values
    model.mag_field = np.zeros((model.size, model.size, model.size, 3), dtype=np.float64)
    model.init_mag_field()
    model.visualize_magnetic_field()
    pass
    


if __name__ == "__main__":
    # test_stamp()
    # test_field_init()
    model = IsingModel(size=6, external_field=(
        0, 0, 0), subdivisions=0, neighbors=1)
    model.lattice = np.ones((6, 6, 6), dtype=np.uint8)
    # Set the second half (rows 3 to 5 along axis 0) to 5
    model.lattice[3:,:,:] = 2
    model.mag_field = np.zeros((model.size, model.size, model.size, 3), dtype=np.float64)
    model.init_mag_field()

    model.visualize_lattice()    
    model.visualize_magnetic_field()

