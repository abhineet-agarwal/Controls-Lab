import numpy as np
import scipy.linalg
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

# Define the system matrices from the MATLAB code
A = np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0.246, -0.18, 0],
    [0, 2.53, -0.246, 0]
])

B = np.array([[0], [0], [0.09], [0.94]])

# Define the system dynamics as a function to simulate
def system_dynamics(t, x, A, B, K):
    u = -np.dot(K, x)
    dxdt = np.dot(A - np.dot(B, K), x) + np.dot(B, u)
    return dxdt

# Define the cost function
def lqr_cost(params):
    Q1, Q2, Q3, Q4, R = params  # Unpack the parameters
    Q = np.diag([Q1, Q2, Q3, Q4])
    R = np.array([[R]])

    # Solve the continuous-time algebraic Riccati equation
    P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    K = np.dot(np.linalg.inv(R), np.dot(B.T, P))
    
    # Set initial conditions near the target angles
    target_theta = 30 * (np.pi / 180)  # 30 degrees in radians
    target_alpha = 3 * (np.pi / 180)   # 3 degrees in radians
    
    initial_theta = target_theta + 1 * (np.pi / 180)  # Slightly perturb from target
    initial_alpha = target_alpha + 0.1 * (np.pi / 180)  # Slightly perturb from target
    initial_dtheta = 0  # No initial angular velocity for theta
    initial_dalpha = 0  # No initial angular velocity for alpha
    x0 = np.array([initial_theta, initial_alpha, initial_dtheta, initial_dalpha])

    # Simulate the system's response over a period of time
    t_span = [0, 5]  # simulate for 5 seconds
    sol = solve_ivp(system_dynamics, t_span, x0, args=(A, B, K), dense_output=True)
    theta = sol.y[0]
    alpha = sol.y[1]

    # Evaluate how well the system meets the target constraints
    theta_deviation = np.max(np.abs(theta - target_theta))  # Max deviation in theta from target
    alpha_deviation = np.max(np.abs(alpha - target_alpha))  # Max deviation in alpha from target

    # Penalize deviations beyond the acceptable limits
    theta_penalty = max(0, theta_deviation - (30 * np.pi / 180))  # ±30 degrees for theta
    alpha_penalty = max(0, alpha_deviation - (3 * np.pi / 180))   # ±3 degrees for alpha

    stability_cost = np.sum(np.abs(np.linalg.eigvals(A - np.dot(B, K))))

    cost = stability_cost + 100 * theta_penalty + 100 * alpha_penalty

    return -cost  # Negate for minimization

# Initial guess for Q and R values
initial_guess = [10, 10, 1, 1, 1]

# Bounds for Q and R (example bounds)
bounds = [(1, 10000), (1, 10000), (1, 10000), (1, 20000), (1 , 1)]

# Perform optimization
result = minimize(lqr_cost, initial_guess, bounds=bounds, method='L-BFGS-B')

print("Optimal Parameters:", result.x)
print("Minimum Cost:", result.fun)
