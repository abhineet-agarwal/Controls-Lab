import numpy as np
import scipy.linalg
from bayes_opt import BayesianOptimization

# Define the system matrices from the MATLAB code
A = np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0.246, -0.18, 0],
    [0, 2.53, -0.246, 0]
])

B = np.array([[0], [0], [0.09], [0.94]])

# Define the cost function
def lqr_cost(Q1, Q2, Q3, Q4, R):
    Q = np.diag([Q1, Q2, Q3, Q4])
    R = np.array([[R]])

    # Solve the continuous-time algebraic Riccati equation
    P = scipy.linalg.solve_continuous_are(A, B, Q, R)

    # Compute the optimal LQR gain matrix
    K = np.dot(np.linalg.inv(R), np.dot(B.T, P))

    # Simulate the system's closed-loop response (example criterion)
    eigenvalues = np.linalg.eigvals(A - np.dot(B, K))
    cost = np.sum(np.abs(eigenvalues))  # Minimize the sum of eigenvalue magnitudes

    return -cost  # Negate cost for minimization

# Define bounds for Q and R (example bounds)
pbounds = {'Q1': (1, 100), 'Q2': (1, 500), 'Q3': (1, 100), 'Q4': (1, 200), 'R': (0.1, 10)}

# Perform Bayesian Optimization
optimizer = BayesianOptimization(
    f=lqr_cost,
    pbounds=pbounds,
    verbose=2,
    random_state=1,
)

optimizer.maximize(
    init_points=10,
    n_iter=50,
)

print("Optimal Parameters:", optimizer.max)
