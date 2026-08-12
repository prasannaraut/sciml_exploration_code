import torch
import torch.nn as nn
import torch.optim as optim


"""
================================================================================
1. WHAT IS A DEEPONET (Deep Operator Network)?
================================================================================
DeepONet is a neural network architecture designed to learn *operators*—mappings
from an entire function space to another function space—rather than traditional
vector-to-vector mappings.

It consists of two main subnetworks:
  - Branch Network: Takes discrete sensor measurements of an input function u(x)
    and converts them into a latent coefficient vector (b).
  - Trunk Network:  Takes continuous evaluation coordinates (y) and converts 
    them into a latent basis vector (t).
  - Output: Computes the dot product (b · t) + bias to predict the output 
    function value G(u)(y) at coordinate y.
================================================================================

================================================================================
2. HOW THIS SPECIFIC IMPLEMENTATION WORKS:
================================================================================
Task: Learn the anti-derivative operator G(u)(y) = \int_0^y u(t) dt

1. Input Sampling (Branch):
   - Input function u(x) is sampled at 20 fixed sensor locations along [0, 1].
   - Branch MLP maps these 20 sensor values -> 32 latent features.

2. Coordinate Query (Trunk):
   - A target evaluation coordinate y ∈ [0, 1] is provided.
   - Trunk MLP maps the 1D coordinate y -> 32 latent features.

3. Combination & Output:
   - Takes the element-wise product of Branch and Trunk vectors, sums them across
     the 32 latent dimensions (dot product), and adds a scalar bias.
   - Predicts the exact continuous integral value at coordinate y without using
     numerical integration algorithms.
================================================================================
"""


class DeepONet(nn.Module):
    def __init__(self, num_sensors: int, latent_dim: int):
        super(DeepONet, self).__init__()

        # Branch network encodes function sensor values [u(x_1), ..., u(x_m)]
        self.branch = nn.Sequential(
            nn.Linear(num_sensors, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

        # Trunk network encodes output query coordinate y
        self.trunk = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

        # Trainable scalar bias
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u_sensors: torch.Tensor, y_coord: torch.Tensor) -> torch.Tensor:
        # u_sensors shape: (batch_size, num_sensors)
        # y_coord shape:   (batch_size, 1)
        b = self.branch(u_sensors)  # (batch_size, latent_dim)
        t = self.trunk(y_coord)  # (batch_size, latent_dim)

        # Inner product across latent feature dimension
        output = torch.sum(b * t, dim=-1, keepdim=True) + self.bias
        return output


# 1. Dataset Generation: G(u)(y) = \int_0^y (c * x) dx = 0.5 * c * y^2
torch.manual_seed(42)
num_samples = 2000
num_sensors = 20

# Sensor locations along x in [0, 1]
x_sensors = torch.linspace(0, 1, num_sensors)

# Random slopes 'c' for functions u(x) = c * x
c = torch.rand(num_samples, 1) * 4.0

# Sensor readings for each function sample: u(x_i)
u_data = c * x_sensors  # Shape: (2000, 20)

# Random query points y in [0, 1]
y_data = torch.rand(num_samples, 1)

# Analytical target: \int_0^y c*t dt = 0.5 * c * y^2
G_target = 0.5 * c * (y_data ** 2)

# 2. Model Initialization
model = DeepONet(num_sensors=num_sensors, latent_dim=32)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# 3. Training Loop
epochs = 1000
for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    predictions = model(u_data, y_data)
    loss = criterion(predictions, G_target)
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d} | MSE Loss: {loss.item():.6f}")

# 4. Inference Test on Unseen Function: u(x) = 3.0 * x
c_test = 3.0
u_test = (c_test * x_sensors).unsqueeze(0)  # Shape: (1, 20)
y_query = torch.tensor([[0.8]])  # Query integral at y = 0.8

with torch.no_grad():
    pred_integral = model(u_test, y_query).item()
    exact_integral = 0.5 * c_test * (0.8 ** 2)

print(f"\n--- Inference Test ---")
print(f"Predicted Integral at y = 0.8 : {pred_integral:.4f}")
print(f"Exact Integral at y = 0.8     : {exact_integral:.4f}")