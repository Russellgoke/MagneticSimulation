from generate_vectors import generate_vectors

class VectorCache:
    def __init__(self, subdivisions=1):
        self.vectors = generate_vectors(subdivisions)

if __name__ == "__main__":
    subdivisions = 1
    vec = VectorCache(subdivisions)
    print(f"A class 1 Geodsic icosahedron with {subdivisions} subdivisions has {len(vec.vectors)} vertices.")
