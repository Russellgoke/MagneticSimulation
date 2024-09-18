from utilities.vector_cache import VectorCache

import numpy as np

class IsingModel:
    def __init__(self, size, subdivisions = 0, neighbors = 4):
        self.size = size
        self.vcache = VectorCache(subdivisions = subdivisions, neighbors=neighbors)
        self.lattice = self.initialize_lattice()

    def initialize_lattice(self):
        lattice = np.random.randint(
            low=0,
            high=self.vcache.vec_num,
            size=(self.size, self.size, self.size),
            dtype=np.uint
        )
        return lattice

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
    model.save_results("data/results.txt")
