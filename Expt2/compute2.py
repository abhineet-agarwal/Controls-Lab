import numpy as np
import matplotlib.pyplot as plt

# Data for each temperature
temp_data = {
    35: np.array([
        [0.000, 0.00], [0.091, 0.05], [0.180, 0.15], [0.253, 0.40],
        [0.307, 0.81], [0.346, 1.37], [0.372, 2.03], [0.392, 2.75],
        [0.400, 3.12], [0.407, 3.51], [0.413, 3.90], [0.425, 4.70],
        [0.438, 5.94], [0.449, 7.18], [0.463, 9.32], [0.473, 11.48],
        [0.482, 13.67]
    ]),
    45: np.array([
        [0.000, 0.00], [0.090, 0.03], [0.177, 0.15], [0.298, 0.84],
        [0.333, 1.41], [0.384, 3.14], [0.401, 4.29], [0.410, 5.08],
        [0.427, 7.10], [0.440, 9.18], [0.449, 11.27], [0.457, 13.39]
    ]),
    55: np.array([
        [0.000, 0.00], [0.089, 0.04], [0.209, 0.31], [0.288, 0.93],
        [0.321, 1.52], [0.344, 2.18], [0.368, 3.28], [0.384, 4.44],
        [0.392, 5.25], [0.408, 7.28], [0.420, 9.36], [0.429, 11.45],
        [0.436, 13.57]
    ]),
    65: np.array([
        [0.000, 0.00], [0.086, 0.07], [0.165, 0.27], [0.226, 0.63],
        [0.270, 1.15], [0.300, 1.79], [0.322, 2.51], [0.338, 3.28],
        [0.351, 4.08], [0.361, 4.91], [0.369, 5.75], [0.385, 7.90],
        [0.397, 10.10], [0.406, 12.33], [0.413, 14.55]
    ])
}

# Create Id-Vd plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for temp, data in temp_data.items():
    plt.plot(data[:, 0], data[:, 1], label=f'{temp}°C')
plt.xlabel('Voltage (V)')
plt.ylabel('Current (mA)')
plt.title('Id-Vd Characteristics')
plt.legend()
plt.grid(True)

# Create ln(Id)-Vd plot
plt.subplot(1, 2, 2)
for temp, data in temp_data.items():
    # Filter out zero current values to avoid log(0)
    non_zero_data = data[data[:, 1] > 0]
    plt.plot(non_zero_data[:, 0], np.log(non_zero_data[:, 1]), label=f'{temp}°C')
plt.xlabel('Voltage (V)')
plt.ylabel('ln(Current)')
plt.title('ln(Id)-Vd Characteristics')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Print some analysis
for temp, data in temp_data.items():
    non_zero_data = data[data[:, 1] > 0]
    slope, intercept = np.polyfit(non_zero_data[:, 0], np.log(non_zero_data[:, 1]), 1)
    print(f"Temperature: {temp}°C")
    print(f"Slope of ln(Id)-Vd: {slope:.2f}")
    print(f"Ideality factor: {1 / (slope * 0.026):.2f}")  # 0.026 is kT/q at room temperature
    print()