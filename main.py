import numpy as np
from ising_model import IsingModel
from vector_cache import VectorCache
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor


def main():
    # Initialize Ising Model
    # J is the exchange interaction and is used as the energy scale
    # T is in units of k_B/ J
    # dipole strength is in units of (mu_0*mu_s^2/4*pi*J*a^3)
    # external field is in units of J 
    vcache = VectorCache(desired_directions=6, d_strength = 1, d_neighbors=3, e_strength=0)
    model = IsingModel((6, 6, 6), vcache, external_field=(0,0,0), wrapping=True, T=0.005)  # Set up params, dimensions, tuple of dimension sizes
    # Run simulation
    # model.run_simulation(200000)
    M, E = model.get_data(gap=20, trials=1000)
    # plt.plot(M)
    plt.plot(E)
    plt.show()
    model.visualize_lattice()
    model.visualize_magnetic_field()
    # MvT1d()

def run_simulation_for_temperature(T, vcache):
    model = IsingModel((6, 6, 1), vcache, external_field=(0, 0, 1), wrapping=False, T=T)
    model.run_simulation(20000)
    avg_magnetization = np.average(model.get_mag_plotz(gap=30, trials=2000))
    return avg_magnetization, model.verify_field_accurate()

def MvT1d():
    N = 40
    T = np.linspace(0.0001, 7, num=N)
    M = np.zeros(N)
    errors = np.zeros(N)
    vcache = VectorCache(desired_directions=50, d_strength=0, d_neighbors=1, e_strength=1)

    # Use ProcessPoolExecutor for parallel execution
    with ProcessPoolExecutor() as executor:
        # Submit each temperature run to the executor
        futures = [executor.submit(run_simulation_for_temperature, T[i], vcache) for i in range(len(T))]

        # Retrieve the results as they complete
        for i, future in enumerate(futures):
            avg_magnetization, error = future.result()
            M[i] = avg_magnetization
            print(f"completed {i}/{N}, {error:.2f}%  error")

    # Plot the results
    plt.plot(T, M)
    plt.xlabel("Temperature (T)")
    plt.ylabel("Average Magnetization (M)")
    plt.show()


if __name__ == "__main__":
    main()
