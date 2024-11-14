import numpy as np
from ising_model import IsingModel
from vector_cache import VectorCache
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor


def main():
    # Initialize Ising Model
    # The energy scale used is J = (mu_0*mu_s^2/4*pi*a^3) and is about 10^-21 J
    # T is in units of k_B/ J
    # dipole strength is in units of (mu_0*mu_s^2/4*pi*J*a^3)
    # external field is in units of J 
    # ext_field_func = const_ext_field((0, 0, 1))
    ext_field_func = oscillating_ext_field(120000)
    vcache = VectorCache(desired_directions=42, d_strength = 0.1, d_neighbors=4, e_strength=1)
    # vcache.visualize_effective_field((0,0,1))
    model = IsingModel((6, 6, 6), vcache, ext_field_func, wrapping=True, T=1.25)  # Set up params, dimensions, tuple of dimension sizes
    # Run simulation
    # model.run_simulation(200000)
    Mx, My, Mz, E = model.get_data(gap=1, trials=240000)
    plt.plot(Mz, label='Mag_z')
    plt.plot(My, label='Mag_y')
    plt.plot(Mx, label='Mag_x')
    # plt.plot(E)
    plt.legend()
    plt.show()
    # model.visualize_lattice()
    # model.visualize_magnetic_field()
    # MvT1d()

def const_ext_field(field):
    def ext_field_func(t):
        return np.array(field, dtype=np.float64)
    return ext_field_func

def z_then_drop(eq_time):
    def ext_field_func(t):
        if t < eq_time:
            return np.array((0, 0, 1), dtype=np.float64)
        else:
            return np.array((0, 0, 0), dtype=np.float64)
    return ext_field_func


def oscillating_ext_field(period):
    omega = 2 * np.pi / period  # Angular frequency
    # Phase shifts for x, y, z components (1/3 of the period apart)
    phase_x = 0
    phase_y = 2 * np.pi / 3  # 120 degrees in radians
    phase_z = 4 * np.pi / 3  # 240 degrees in radians
    def ext_field_func(t):
        # Calculate the field components
        field_x = np.sin(omega * t + phase_x)
        field_y = np.sin(omega * t + phase_y)
        field_z = np.sin(omega * t + phase_z)
        return np.array([field_x, field_y, field_z], dtype=np.float64)
    return ext_field_func


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
