import numpy as np
from scipy.linalg import solve_continuous_are, pinv

# Define the system parameters
M_p = 0.027  # Pendulum mass (kg)
l_p = 0.153  # Pendulum length (m)
L_p = 0.191  # Arm length (m)
r = 0.08260  # Wheel radius (m)

J_m = 3e-5   # Motor inertia (kg-m^2)
M_arm = 0.028  # Arm mass (kg)
g = 9.810    # Gravitational acceleration (m/s^2)
J_eq = 1.23e-4  # Equivalent inertia (kg-m^2)
J_p = 1.1e-4  # Pendulum inertia (kg-m^2)
B_eq = 0     # Equivalent damping (N-m/(rad/s))
B_p = 0      # Pendulum damping (N-m/(rad/s))
R_m = 3.3    # Motor resistance (ohm)
K_t = 0.02797  # Torque constant (N-m)
K_m = 0.02797  # Motor constant (N-m)

# Construct the state-space matrices
A = np.zeros((4, 4))
A[0, 3] = 1
A[1, 3] = 1
A[2, 1] = (r * M_p**2 * l_p**2 * g) / (J_p * J_eq + M_p * l_p**2 * J_eq + J_p * M_p * r**2)
A[2, 3] = -(K_t * K_m * (J_p + M_p * l_p**2)) / ((J_p * J_eq + M_p * l_p**2 * J_eq + J_p * M_p * r**2) * R_m)
A[3, 1] = (M_p * l_p * g * (J_eq + M_p * r**2)) / (J_p * J_eq + M_p * l_p**2 * J_eq + J_p * M_p * r**2)
A[3, 3] = -(M_p * l_p * K_t * r * K_m) / ((J_p * J_eq + M_p * l_p**2 * J_eq + J_p * M_p * r**2) * R_m)

B = np.zeros((4, 1))
B[2, 0] = (K_t * (J_p + M_p * l_p**2)) / ((J_p * J_eq + M_p * l_p**2 * J_eq + J_p * M_p * r**2) * R_m)
B[3, 0] = (M_p * l_p * K_t * r) / ((J_p * J_eq + M_p * l_p**2 * J_eq + J_p * M_p * r**2) * R_m)

C = np.eye(4)
D = np.zeros((4, 1))

# Design the LQR controller
Q = np.diag([22, 500, 20, 150])
R = 1
N = np.zeros((4, 1))

K = solve_continuous_are(A, B, Q, R)
K = -pinv(B.T @ K + R) @ B.T @ K

# Print the optimal control gains
print(f"float k1 = {K[0,0]:.4f};")
print(f"float k2 = {K[1,0]:.4f};")
print(f"float k3 = {K[2,0]:.4f};")
print(f"float k4 = {K[3,0]:.4f};")

import control

# Extract the transfer function
b, a = control.ss2tf(A, B, C, D)
H = control.tf(b, a)
print("Transfer function:")
print(H)

import numpy as np
from scipy.signal import step

def evaluate_performance(Q, R):
    # Compute the LQR controller gain
    K = solve_continuous_are(A, B, Q, R)
    K = -pinv(B.T @ K + R) @ B.T @ K

    # Construct the closed-loop state-space model
    A_cl = A - B @ K
    B_cl = B
    C_cl = C
    D_cl = D

    # Simulate the step response
    t, y = step(control.ss(A_cl, B_cl, C_cl, D_cl))

    # Compute the performance metrics
    theta_max = np.max(np.abs(y[:, 0]))
    alpha_max = np.max(np.abs(y[:, 1]))

    # Check if the requirements are satisfied
    if theta_max <= 30 and alpha_max <= 3:
        return True, theta_max, alpha_max
    else:
        return False, theta_max, alpha_max

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
        
        # Define the search space for Q and R
Q_min, Q_max = 0, 1000
R_min, R_max = 0, 1000
        
        # Create the Gaussian Process Regressor
kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        
        # Optimize the LQR controller parameters
X = []
y = []
        
while len(X) < 100:
    q1 = np.random.uniform(Q_min, Q_max)
    q2 = np.random.uniform(Q_min, Q_max)
    q3 = np.random.uniform(Q_min, Q_max)
    q4 = np.random.uniform(Q_min, Q_max)
    r = np.random.uniform(R_min, R_max)
    Q = np.diag([q1, q2, q3, q4])

    satisfied, theta_max, alpha_max = evaluate_performance(Q, r)
    X.append([q1, q2, q3, q4, r])
    y.append([theta_max, alpha_max, int(satisfied)])
        
X = np.array(X)
y = np.array(y)

gpr.fit(X, y)

# Find the optimal LQR controller parameters
q1_opt, q2_opt, q3_opt, q4_opt, r_opt = gpr.predict([[0, 0, 0, 0, 0]], return_std=False)[0]
Q_opt = np.diag([q1_opt, q2_opt, q3_opt, q4_opt])

print(f"Optimal Q: {Q_opt}")
print(f"Optimal R: {r_opt}")