import numpy as np
from generate_vectors import generate_vectors
from generate_mag_field_contribution import calculate_magnetic_field

class VectorCache:
    def __init__(self, subdivisions=1, neighbors = 2):
        self.vectors = generate_vectors(subdivisions)
        vec_num = len(self.vectors)
        grid_size = 2 * neighbors + 1
        self.dipole_contributions = np.zeros((vec_num, grid_size, grid_size, grid_size, 3), dtype=np.float64)
        for i in range(vec_num):
            self.dipole_contributions[i] += calculate_magnetic_field(self.vectors[i], mu = 1, grid_size = grid_size)

if __name__ == "__main__":
    subdivisions = 1
    vec = VectorCache(subdivisions, neighbors = 4)
    print(f"A class 1 Geodsic icosahedron with {subdivisions} subdivisions has {len(vec.vectors)} vertices.")
    print(f"Dimensions of the dipole_contribution data is {vec.dipole_contributions.shape}")
