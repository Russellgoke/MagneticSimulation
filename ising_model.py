from utilities.vector_cache import VectorCache
import matplotlib.pyplot as plt
from copy import deepcopy

import numpy as np


class IsingModel:
    def __init__(self, size=5, wrapping = False, external_field=(0, 0, 0), subdivisions=0, neighbors=4, T = 1.0):
        self.size = size
        self.external_field = external_field
        self.wrapping = wrapping
        self.vcache = VectorCache(
            subdivisions=subdivisions, neighbors=neighbors)
        self.lattice = self.init_lattice()
        self.mag_field = np.full((self.size[0], self.size[1], self.size[2], 3),
                                 external_field, dtype=np.float64)
        self.init_mag_field()
        self.T = T  # Temperature (in units of k_B)

    def init_lattice(self):
        lattice = np.random.randint(
            low=0,
            high=self.vcache.num_vec,
            size=self.size,
            dtype=np.uint8  # TODO make work with larger too
        )
        return lattice

    def init_mag_field(self):
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                for z in range(self.size[2]):
                    vec_num = self.lattice[x][y][z]
                    self.stamp_onto_field(
                        (x, y, z), self.vcache.dipole_contributions[vec_num])

    def stamp_onto_field(self, center, stamp):
        """
        Stamps a smaller 3D array (stamp) onto a larger 3D magnetic field array at a specified center position.
        Wrapping around the boundaries occurs only if the 'wrapping' parameter is True.
        For wrapping=False, the function uses slicing for better performance.
        
        Parameters:
        - center (tuple): A tuple (x, y, z) indicating the center position on the magnetic field where the stamp will be applied.
        - stamp (numpy.ndarray): A smaller 3D array representing the values to add to the magnetic field.
        - wrapping (bool): If True, the stamp will wrap around the boundaries of the magnetic field array.
                        If False, the stamp will not wrap and will adjust for boundaries.
        """
        radius = stamp.shape[0] // 2  # Assuming the stamp is cubic with odd dimensions

        x_start = center[0] - radius
        y_start = center[1] - radius
        z_start = center[2] - radius

        x_end = center[0] + radius + 1
        y_end = center[1] + radius + 1
        z_end = center[2] + radius + 1

        if self.wrapping:
            # Wrap indices using modulo operation
            x_indices = np.arange(x_start, x_end) % self.size[0]
            y_indices = np.arange(y_start, y_end) % self.size[1]
            z_indices = np.arange(z_start, z_end) % self.size[2]

            # Use np.ix_ to create a 3D grid of indices
            ix, iy, iz = np.ix_(x_indices, y_indices, z_indices)

            # Stamp the entire stamp onto the magnetic field
            self.mag_field[ix, iy, iz] += stamp
        else:
            # Adjust indices to avoid going out of bounds
            # Calculate the overlapping region between the stamp and the field
            x_start_field = max(x_start, 0)
            y_start_field = max(y_start, 0)
            z_start_field = max(z_start, 0)

            x_end_field = min(x_end, self.size[0])
            y_end_field = min(y_end, self.size[1])
            z_end_field = min(z_end, self.size[2])

            # Corresponding indices in the stamp checking if the min or max affected it
            x_start_stamp = x_start_field - x_start
            y_start_stamp = y_start_field - y_start
            z_start_stamp = z_start_field - z_start

            x_end_stamp = x_end_field - x_start
            y_end_stamp = y_end_field - y_start
            z_end_stamp = z_end_field - z_start

            # Use slicing for better performance
            self.mag_field[x_start_field:x_end_field,
                        y_start_field:y_end_field,
                        z_start_field:z_end_field] += stamp[x_start_stamp:x_end_stamp,
                                                            y_start_stamp:y_end_stamp,
                                                            z_start_stamp:z_end_stamp]


    def run_simulation(self, iterations):
        for _ in range(iterations):
            self.update_lattice()

    def update_lattice(self):
        # Pick a random site
        x = np.random.randint(0, self.size[0])
        y = np.random.randint(0, self.size[1])
        z = np.random.randint(0, self.size[2])

        # Store the old spin index and vector
        old_spin_idx = self.lattice[x, y, z]
        old_spin_vec = self.vcache.vectors[old_spin_idx]
        old_energy = -np.dot(old_spin_vec, self.mag_field[x][y][z])

        # Propose a new spin index (ensure it's different from the current one)
        num_spins = len(self.vcache.vectors)
        new_spin_idx = np.random.randint(0, num_spins)
        while new_spin_idx == old_spin_idx:
            new_spin_idx = np.random.randint(0, num_spins)
        new_spin_vec = self.vcache.vectors[new_spin_idx]
        new_energy = -np.dot(new_spin_vec, self.mag_field[x][y][z])

        # Total energy change
        delta_E = new_energy - old_energy

        # Decide whether to accept the change
        if delta_E <= 0:
            accept = True
        else:
            probability = np.exp(-delta_E / self.T)
            accept = np.random.rand() < probability

        if accept:
            # Update the lattice
            self.lattice[x, y, z] = new_spin_idx

            # Update the magnetic field
            # Remove the old spin's contribution and add the new spin's contribution
            old_stamp = self.vcache.dipole_contributions[old_spin_idx]
            new_stamp = self.vcache.dipole_contributions[new_spin_idx]
            delta_stamp =  new_stamp - old_stamp
            self.stamp_onto_field((x, y, z), delta_stamp)

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

        Y, X, Z = np.meshgrid(np.arange(self.size[1]), np.arange(
            self.size[0]), np.arange(self.size[2]))

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
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                for z in range(self.size[2]):
                    lattice_vecs[x][y][z] = self.vcache.vectors[self.lattice[x][y][z]]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        Y, X, Z = np.meshgrid(np.arange(self.size[1]), np.arange(
            self.size[0]), np.arange(self.size[2]))

        u = lattice_vecs[..., 0]  # X component of B
        v = lattice_vecs[..., 1]  # Y component of B
        w = lattice_vecs[..., 2]  # Z component of B

        ax.quiver(X, Y, Z, u, v, w, length=0.5)

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()

    def verify_field_accurate(self):
        old = deepcopy(self.mag_field)
        self.mag_field = np.full((self.size[0], self.size[1], self.size[2], 3),
                            self.external_field, dtype=np.float64)
        self.init_mag_field()
        diff = self.mag_field - old
        max_diff_magnitude = np.linalg.norm(diff, axis=-1).max()
        mag_field_magnitudes = np.linalg.norm(self.mag_field, axis=-1)
        mean_mag_field_magnitude = mag_field_magnitudes.mean()
        error = max_diff_magnitude / mean_mag_field_magnitude
        return error
        


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
    model = IsingModel(size=(5,5,5), external_field=(
        0, 0, 0), subdivisions=0, neighbors=0)
    # these vectors form a 4 by 3 in space to try and help visualize, num_vec = 12
    test_vecs = [[1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [1, 0, -1], [0, 1, -1], [-1, 0, -1], [0, -1, -1]]
    model.vcache.dipole_contributions = np.reshape(test_vecs, (12, 1, 1, 1, 3))
    # reset and re init mag field with fake values
    model.mag_field = np.zeros((model.size[0], model.size[1], model.size[2], 3), dtype=np.float64)
    model.init_mag_field()
    model.visualize_magnetic_field()
    
    


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

