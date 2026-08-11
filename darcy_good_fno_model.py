# adding positional embedding, lifting layers, domain padding, skip connections in FNO block, domain unpadding and projection layer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from neuralop.data.datasets import load_darcy_flow_small

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys

############################
n_train = 500
# batch_size = 16
n_tests = 30
train_file_16 = "darcy_train_16.pt"
train_file_32 = "darcy_train_32.pt"
test_file_16 = "darcy_test_16.pt"
test_file_32 = "darcy_test_16.pt"

input_resolution = 16
output_resolution = 32


plot_index_to_check = 1





# --------------------------------------------------------
# 1. Fourier Layer Definition
# --------------------------------------------------------
class SpectralConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                dtype=torch.cfloat,
            )
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                dtype=torch.cfloat,
            )
        )

    def forward(self, x):
        batchsize = x.shape[0]

        # Compute Fourier coefficients
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        out_ft[:, :, : self.modes1, : self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, : self.modes1, : self.modes2],
            self.weights1,
        )

        out_ft[:, :, -self.modes1 :, : self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, -self.modes1 :, : self.modes2],
            self.weights2,
        )

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


# --------------------------------------------------------
# 2. FNO Block (Spectral Conv + Spatial Skip Connection)
# --------------------------------------------------------
class FNOBlock2d(nn.Module):

    def __init__(self, channels, modes1, modes2):
        super(FNOBlock2d, self).__init__()
        # Global frequency-domain convolution
        self.conv = SpectralConv2d(channels, channels, modes1, modes2)
        # Local spatial-domain skip connection (1x1 conv)
        self.w = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        # Residual addition before non-linear activation
        return F.gelu(self.conv(x) + self.w(x))


# --------------------------------------------------------
# 3. Complete FNO Architecture
# --------------------------------------------------------
class FNO2d(nn.Module):

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        modes1=8,
        modes2=8,
        width=32,
        num_layers=4,
        padding=8,
        out_res = output_resolution,
    ):
        super(FNO2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = padding
        self.out_res = out_res

        # 1. Lifting Layer: 2-layer MLP (in_channels + 2 spatial coords -> width)
        self.lifting = nn.Sequential(
            nn.Conv2d(in_channels + 2, self.width // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.width // 2, self.width, kernel_size=1),
        )

        # 2. FNO Backbone: Stack of spectral blocks with skip connections
        self.fno_blocks = nn.ModuleList(
            [
                FNOBlock2d(self.width, self.modes1, self.modes2)
                for _ in range(num_layers)
            ]
        )

        # 3. Projection Layer: 2-layer MLP (width -> 128 -> out_channels)
        self.projection = nn.Sequential(
            nn.Conv2d(self.width, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(128, out_channels, kernel_size=1),
        )

    def get_grid(self, shape, device):
        """Generates 2D normalized positional grid [0, 1] x [0, 1]."""
        batchsize, _, size_x, size_y = shape
        gridx = torch.linspace(0, 1, size_x, device=device)
        gridy = torch.linspace(0, 1, size_y, device=device)
        gridx, gridy = torch.meshgrid(gridx, gridy, indexing="ij")

        gridx = gridx.reshape(1, 1, size_x, size_y).repeat(batchsize, 1, 1, 1)
        gridy = gridy.reshape(1, 1, size_x, size_y).repeat(batchsize, 1, 1, 1)

        return torch.cat((gridx, gridy), dim=1)

    def forward(self, x):
        # Step 1: Positional Embedding (Concatenate normalized x, y coordinates along channel dim)
        x = F.interpolate(x, size=(self.out_res, self.out_res), mode='bicubic', align_corners=True)

        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=1)

        # Step 2: Lifting Layer (Lift channel dimension from (C_in + 2) to width)
        x = self.lifting(x)

        # Step 3: Domain Padding (Pad boundaries to handle non-periodic conditions)
        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])

        # Step 4: FNO Blocks (Fourier convolutions + skip connections)
        for block in self.fno_blocks:
            x = block(x)

        # Step 5: Domain Unpadding (Remove extra padded boundary pixels)
        if self.padding > 0:
            x = x[..., : -self.padding, : -self.padding]

        # Step 6: Projection Layer (Project back to desired output channels)
        x = self.projection(x)
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
    model = FNO2d(in_channels=1, out_channels=1, modes1=8, modes2=8, width=64, num_layers=8, padding=4).to(device)


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
    data_index = plot_index_to_check
    ## checking the data
    yt = y_test[0].cpu().numpy()
    yp = y_pred[0].cpu().numpy()
    channel_idx = 0


    plot_predicted_original(yt,yp,0)