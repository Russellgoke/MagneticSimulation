import numpy as np
import matplotlib.pyplot as plt

# mu is mu/4pi * magnitude of m, magnitude of m should be 1

def calc_simp_mag_field(m, mu=1, grid_size=3):
    half_grid = grid_size // 2

    # Initialize a grid_size x grid_size x grid_size x 3 array to store the magnetic field vectors
    B_vectors = np.full((grid_size, grid_size, grid_size, 3), m*mu)
    B_vectors[half_grid, half_grid, half_grid] = 0

    return B_vectors


def calculate_magnetic_field(m, mu=1, grid_size=5):
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


def visualize_magnetic_field(B_vectors):
    grid_size = B_vectors.shape[0]
    half_grid = grid_size // 2

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Prepare the grid and vectors, this order ensures the vec[x][y][z] pattern
    y, x, z = np.meshgrid(
        np.arange(-half_grid, half_grid + 1),
        np.arange(-half_grid, half_grid + 1),
        np.arange(-half_grid, half_grid + 1)
    )

    u = B_vectors[..., 0]  # X component of B
    v = B_vectors[..., 1]  # Y component of B
    w = B_vectors[..., 2]  # Z component of B

    ax.quiver(x, y, z, u, v, w, length=0.5, normalize=True)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Magnetic Field Vectors')

    plt.show()


if __name__ == "__main__":
    m_dipole = np.array([0, 0, 1])  # Magnetic dipole moment
    # magnetic_field_grid = calculate_magnetic_field(m_dipole, mu=1, grid_size=5)
    magnetic_field_grid = calc_simp_mag_field(m_dipole, mu=1, grid_size=3)
    # Visualize the magnetic field vectors
    visualize_magnetic_field(magnetic_field_grid)
    pass
