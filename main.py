import numpy as np
from ising_model import IsingModel
import matplotlib.pyplot as plt

def main():
    # Initialize Ising Model
    # MvT1d()

    model = IsingModel((10, 10, 10), external_field=(0,0,0), wrapping=False, neighbors = 1, T=0.2, desired_directions=6, dipole=False)  # Set up params, dimensions, tuple of dimension sizes
    # Run simulation
    # model.run_simulation(200000)
    M = model.get_mag_plot(gap=30, trials=10000)
    plt.plot(M)
    plt.show()
    model.visualize_lattice()
    pass

def MvT1d():
    T = np.linspace(0.0001, 4, num = 30)
    M = np.zeros(30)
    for i in range(len(T)):
        print(f"{i}/{len(T)}")
        model = IsingModel((20, 20, 1), wrapping=False,  neighbors = 1, T=T[i])  # Set up params, dimensions, tuple of dimension sizes
        # Run simulation
        model.run_simulation(200000)
        M[i] = np.average(model.get_mag_plot(gap=30, trials=500))

    plt.plot(T, M)
    plt.show()

    print(f"{model.verify_field_accurate()*100:.2f}% error")

if __name__ == "__main__":
    main()
