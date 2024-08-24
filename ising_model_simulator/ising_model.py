from .utilities.vector_cache import VectorCache

class IsingModel:
    def __init__(self, x_dim, y_dim, z_dim, subdivisions = 1, neighbors = 4):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.vcache = VectorCache(subdivisions = subdivisions, neighbors=neighbors)
        self.lattice = self.initialize_lattice()

    def initialize_lattice(self):
        # TODO Implement
        return []

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

if __name__ == "__main__":
    model = IsingModel(4, 3, 2)
    model.run_simulation(100)
    model.save_results("results.txt")
