import numpy as np

def sigmoid(x):
    """Standard sigmoid function."""
    return 1 / (1 + np.exp(-x))

# Set seed for reproducibility
np.random.seed(42)

# --- Vectors ---
# Center word vector
v_c = np.array([0.1, 0.2, 0.3])

# True outside word vector (positive sample)
u_o = np.array([0.3, 0.2, 0.1])

# Negative samples
u_k1 = np.array([-0.2, -0.1, 0.0])
u_k2 = np.array([-0.1, 0.0, 0.1])

# Learning rate
alpha = 0.05

# --- Step 1: Compute dot products ---
dot_pos = np.dot(v_c, u_o)
dot_neg1 = np.dot(v_c, u_k1)
dot_neg2 = np.dot(v_c, u_k2)

# --- Step 2: Apply sigmoid ---
score_pos = sigmoid(dot_pos)
score_neg1 = sigmoid(-dot_neg1)
score_neg2 = sigmoid(-dot_neg2)

# --- Step 3: Compute loss ---
loss = -np.log(score_pos) - np.log(score_neg1) - np.log(score_neg2)

# --- Step 4: Compute gradient w.r.t. center word vector ---
grad_vc = (sigmoid(dot_pos) - 1) * u_o + sigmoid(dot_neg1) * u_k1 + sigmoid(dot_neg2) * u_k2

# --- Step 5: Update center word vector using SGD ---
v_c_updated = v_c - alpha * grad_vc

# --- Output ---
print("Loss:", round(loss, 6))
print("Gradient:", grad_vc)
print("Updated v_c:", v_c_updated)
