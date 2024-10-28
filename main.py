from ising_model import IsingModel

def main():
    # Initialize Ising Model
    model = IsingModel(2, subdivisions = 0, neighbors = 1, T=100)  # Set up params, dimensions, tuple of dimension sizes

    # Run simulation
    model.run_simulation(100000)  # Example number of iterations

    print(model.verify_field_accurate())
    # model.visualize_lattice()
    # model.visualize_magnetic_field()
    # Save results
    # model.save_results("data/results.txt")

if __name__ == "__main__":
    main()
