import numpy as np
from ising_model import IsingModel
from utilities.vector_cache import VectorCache
import matplotlib.pyplot as plt

def main():
    # Initialize Ising Model
    # J is the exchange interaction and is used as the energy scale
    # T is in units of k_B/ J
    # dipole strength is in units of (mu_0*mu_s^2/4*pi*J*a^3)
    # external field is in units of J 
    vcache = VectorCache(desired_directions=2, d_strength = 1, d_neighbors=1, exchange=True)
    model = IsingModel((10, 1, 1), vcache, external_field=(0,0,2.1), wrapping=False, T=0.0001)  # Set up params, dimensions, tuple of dimension sizes
    # Run simulation
    # model.run_simulation(200000)
    M = model.get_mag_plotz(gap=1, trials=1000)
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
