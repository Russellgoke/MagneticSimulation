from ising_model import IsingModel

def main():
    # Initialize Ising Model
    model = IsingModel(2, subdivisions = 0, neighbors = 1)  # Set up params, dimensions, tuple of dimension sizes

    # Run simulation
    model.run_simulation(1000)  # Example number of iterations

    # Save results
    model.save_results("data/results.txt")

if __name__ == "__main__":
    main()
