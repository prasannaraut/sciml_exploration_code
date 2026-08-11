# fno has only the spectral convolution model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from neuralop.data.datasets import load_darcy_flow_small

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import sys

############################
n_train = 500
batch_size = 16
n_tests = 30
train_file_16 = "darcy_train_16.pt"
train_file_32 = "darcy_train_32.pt"
test_file_16 = "darcy_test_16.pt"
test_file_32 = "darcy_test_32.pt"

input_resolution = 16
output_resolution = 32




# --------------------------------------------------------
# 1. Fourier Layer Definition
# --------------------------------------------------------
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Number of Fourier modes to multiply
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(
            in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(
            in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]

        # Compute Fourier coefficients
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)

        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.weights1)

        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


# --------------------------------------------------------
# 2. FNO Model Architecture
# --------------------------------------------------------
class FNO2d(nn.Module):
    def __init__(self, modes1=8, modes2=8, width=32, out_res = 16):
        super(FNO2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.out_res = out_res

        # Lift the input channel to higher dimensional feature space
        self.p = nn.Conv2d(1, self.width, 1)

        # Spectral convolution layers
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        # Skip connections (1x1 convolutions)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        # Project back to 1 output channel
        self.q = nn.Conv2d(self.width, 1, 1)

    def forward(self, x):
        # 1. Upsample input (16x16) to match the target grid (32x32)
        x = F.interpolate(x, size=(self.out_res, self.out_res), mode='bicubic', align_corners=True)

        # 2. Lift
        x = self.p(x)

        # 3. Apply FNO blocks
        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = F.gelu(self.conv3(x) + self.w3(x))

        # 4. Project
        x = self.q(x)
        return x


# --------------------------------------------------------
# 3. Training Loop & Prediction
# --------------------------------------------------------
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_train_x = torch.load(sys.path[0] + "\\data\\darcy_train_" + str(input_resolution) + ".pt")
    data_train_y = torch.load(sys.path[0] + "\\data\\darcy_train_" + str(output_resolution) + ".pt")

    data_test = torch.load(sys.path[0] + "\\data\\darcy_train_" + str(input_resolution) + ".pt")

    x_train = data_train_x["x"].float()[0:n_train].unsqueeze(1).to(device)
    y_train = data_train_y["y"][0:n_train].unsqueeze(1).to(device)


    x_test = data_test["x"].float()[0:n_train].unsqueeze(1).to(device)
    y_test = data_test["y"][0:n_train].unsqueeze(1).to(device)


    # Initialize model, loss function, and optimizer
    # (modes1 and modes2 must be <= 16 since max target dim is 32)
    model = FNO2d(modes1=8, modes2=8, width=128, out_res = output_resolution).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()

    epochs = 2000
    print("Starting Training...")

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        predictions = model(x_train)

        # Compute loss
        loss = criterion(predictions, y_train)

        # Backward pass
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}")

    print("Training Complete!\n")

    # --------------------------------------------------------
    # 4. Prediction on a new tensor
    # --------------------------------------------------------
    model.eval()

    # Create a new, unseen input tensor of the same input shape

    with torch.no_grad():
        y_pred = model(x_test)

    print(f"New Input shape: {x_test.shape}")
    print(f"Prediction shape: {y_pred.shape}")




    def plot_predicted_original(yt, yp, channel_idx):
        # Extract the 2D grids and convert to numpy arrays
        yt_grid = yt[channel_idx, :, :]
        yp_grid = yp[channel_idx, :, :]

        # 3. Initialize the figure with 1 row and 2 columns
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Data yt",
                            "Data yp")
        )

        # 4. Add the heatmap for tensor 'x' (Column 1)
        fig.add_trace(
            go.Heatmap(
                z=yt_grid,
                colorscale='Viridis',
                # Offset the colorbar so it doesn't overlap with the second plot
                colorbar=dict(title="yt values", x=0.45, len=0.75)
            ),
            row=1, col=1
        )

        # 5. Add the heatmap for tensor 'y' (Column 2)
        fig.add_trace(
            go.Heatmap(
                z=yp_grid,
                colorscale='Viridis',
                colorbar=dict(title="yp values", x=1.0, len=0.75)
            ),
            row=1, col=2
        )

        # 6. Update layout to ensure aspect ratios are square and titles look good
        fig.update_layout(
            title_text="Heatmap Comparison of Data yt and yp",
            height=500,
            width=1000,
            showlegend=False
        )

        # Optional: Reverse the Y-axes if you want them to match typical image coordinates (0,0 at top-left)
        fig.update_yaxes(autorange="reversed", row=1, col=1)
        fig.update_yaxes(autorange="reversed", row=1, col=2)

        # Ensure grid cells remain perfectly square
        fig.update_xaxes(scaleanchor="y", scaleratio=1, row=1, col=1)
        fig.update_xaxes(scaleanchor="y", scaleratio=1, row=1, col=2)

        # 7. Render the plot
        fig.show()


    ## checking the data
    data_index = 0
    ## checking the data
    yt = y_test[0].cpu().numpy()
    yp = y_pred[0].cpu().numpy()
    channel_idx = 0


    plot_predicted_original(yt,yp,0)
