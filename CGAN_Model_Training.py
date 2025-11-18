# Make sure have the ready made excel file in name: TF CGAN.xlsx

# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.stats import kurtosis
from scipy.stats import skew
from sklearn.metrics import r2_score
import math
import warnings
import csv
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import autograd
import torch.optim as optim
from torch.autograd import Variable
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

df = pd.read_excel('filePath')


input_cols = df.columns['Input Columns in datafile']
target_cols = df.columns['Target Columns in datafile']

# Converting target columns into log scale
X_real = df[input_cols].copy()
for col in target_cols:
    X_real[col] = np.log10(df[col])

X_real.replace([np.inf, -np.inf], np.nan, inplace=True)
X_real.dropna(inplace=True)

# Splitting into train, validation & test sets
X_train, X_temp = train_test_split(X_real, test_size=0.3, random_state=42)
X_val, X_test = train_test_split(X_temp, test_size=0.5, random_state=42)

# Min-Max Scaler for Input features
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(X_train[input_cols])

X_train_scaled = X_train.copy()
X_train_scaled[input_cols] = scaler.transform(X_train[input_cols])

X_val_scaled = X_val.copy()
X_val_scaled[input_cols] = scaler.transform(X_val[input_cols])

X_test_scaled = X_test.copy()
X_test_scaled[input_cols] = scaler.transform(X_test[input_cols])

"""# **Model Building - Generator & Descriminator**"""

# Define Discriminator
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(len(target_cols) + len(input_cols), 32), # Sum of target_cols and input_cols
            nn.ELU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ELU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x, c):
        c = c.view(c.size(0), -1)
        x = torch.cat((x, c), dim=1)
        return self.model(x.float())

# Generate real, fake samples
def generate_real_samples(df_real, n):
    sample = df_real.sample(n)
    cond = torch.tensor(sample[input_cols].values, dtype=torch.float32).to(device)
    target = torch.tensor(sample[target_cols].values, dtype=torch.float32).to(device)
    return target, cond

def generate_latent_points(latent_dim, n_samples, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randn((n_samples, latent_dim), device=device)

def generate_fake_samples(generator, latent_dim, n_samples, conditions, seed=None):
    z = generate_latent_points(latent_dim, n_samples, seed=seed)
    conditions = torch.tensor(conditions, dtype=torch.float32).view(n_samples, -1).to(device)
    return generator(z, conditions), conditions


def predict(conditions_test, generator, latent_dim=5, mode="mean", n_samples=10, seed=None):
    """
    Predict using generator with different z handling modes.

    Args:
        conditions_test: numpy array or tensor of condition inputs
        generator: trained generator model
        latent_dim: size of latent space (default=5)
        mode: "zero" → use z=0
              "mean" → average predictions over n_samples random z's
        n_samples: number of random z's to average if mode="mean"
        seed: seed for reproducibility when mode="mean"
    """
    generator.eval()
    with torch.no_grad():
        c = torch.tensor(conditions_test, dtype=torch.float32).to(device)

        if mode == "zero":
            # z = all zeros → deterministic output
            z = torch.zeros((c.shape[0], latent_dim), device=device)
            preds = generator(z, c)

        elif mode == "mean":
            # average predictions over multiple random z's
            preds_list = []
            for _ in range(n_samples):
                z = generate_latent_points(latent_dim, c.shape[0], seed=seed + _) # Use different seed for each sample
                preds_list.append(generator(z, c))
            preds = torch.mean(torch.stack(preds_list), dim=0)

        else:
            raise ValueError("mode must be 'zero' or 'mean'")

        # Concatenate input conditions + generated predictions
        input_values = c
        preds = torch.cat((input_values, preds), dim=1).cpu().numpy()

        return preds

def generate_fixed_latent_points(latent_dim, n_samples, seed=42):
    """Generates a fixed latent vector for reproducible results."""
    torch.manual_seed(seed)
    return torch.randn((n_samples, latent_dim), device=device)



"""# **Training Model**"""

def train(g_model, d_model, data, val_data, latent_dim=5, n_epochs=1000, n_batch=128, best_r2=-1e6):
    batch_per_epoch = max(1, len(data) // n_batch)
    half_batch = n_batch // 2

    # Reshape y_real and y_fake to have two dimensions
    y_real = torch.ones((half_batch, 1), device=device) # 32
    y_fake = torch.zeros((half_batch, 1), device=device) # 32


    for epoch in range(n_epochs):
        for _ in range(batch_per_epoch):
            real_X, real_c = generate_real_samples(data, half_batch) # 27, 6
            fake_X, fake_c = generate_fake_samples(g_model, latent_dim, half_batch, real_c.cpu().numpy()) #

            d_model.zero_grad()
            loss_real = criterion(d_model(real_X, real_c), y_real)
            loss_fake = criterion(d_model(fake_X.detach(), fake_c), y_fake)
            d_loss = loss_real + loss_fake
            d_loss.backward()
            dis_optimizer.step()

            g_model.zero_grad()
            g_loss = criterion(d_model(fake_X, fake_c), y_real)
            g_loss.backward()
            gen_optimizer.step()

        if epoch % 100 == 0:
            conditions_val = val_data[input_cols].to_numpy()
            preds = predict(conditions_val, g_model, latent_dim=latent_dim, seed=epoch) # Pass latent_dim and epoch as seed to predict
            true_vals = val_data[input_cols.tolist() + target_cols.tolist()].to_numpy()
            preds = preds  # Already scaled + log-transformed

            curr_r2 = r2_score(true_vals[:, len(input_cols):], preds[:, len(input_cols):])
            print(f"Epoch {epoch}: R2 = {curr_r2:.4f}, Best R2 = {best_r2:.4f}, Generator Loss = {g_loss}, Discriminator Loss = {d_loss}")

            if curr_r2 > best_r2:
                torch.save(g_model.state_dict(), 'best_generator.pth')
                torch.save(d_model.state_dict(), 'best_discriminator.pth')
                best_r2 = curr_r2

            with open('results.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, best_r2, curr_r2, d_loss.item(), g_loss.item()])

# Define Generator
class Generator(nn.Module):
    def __init__(self, latent_dim, input_cols_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + input_cols_dim, 48),
            nn.ELU(0.2),
            nn.Dropout(0.2),
            nn.Linear(48, 36),
            nn.ELU(0.2),
            nn.Dropout(0.2),
            nn.Linear(36, output_dim) # Use output_dim for the final layer
        )
    def forward(self, z, c):
        c = c.view(c.size(0), -1)
        x = torch.cat((z, c), dim=1)
        return self.model(x.float())

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

latent_dim = 15
generator = Generator(latent_dim, len(input_cols), len(target_cols)).to(device)
discriminator = Discriminator().to(device)

generator_params = count_parameters(generator)
discriminator_params = count_parameters(discriminator)
total_params = generator_params + discriminator_params

print(f"Number of parameters in Generator: {generator_params}")
print(f"Number of parameters in Discriminator: {discriminator_params}")
print(f"Total parameters in Generator and Discriminator: {total_params}")

criterion = nn.BCELoss()

gen_optimizer = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
dis_optimizer = optim.Adam(discriminator.parameters(), lr=5e-4, betas=(0.5, 0.999))

train(generator, discriminator, X_train_scaled, X_val_scaled, latent_dim=latent_dim)

from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Compute R² on test set (only PSa columns)
true_values = X_test_scaled[input_cols.tolist() + target_cols.tolist()].to_numpy()
conditions_test = X_test_scaled[input_cols].to_numpy()

# Generate 1000 stochastic predictions
outputs = [predict(conditions_test, generator, latent_dim=latent_dim, seed=_) for _ in range(1000)]
mean_array = np.mean(outputs, axis=0)

# R² before slicing
r_squared_test = r2_score(true_values, mean_array)
print("R² (full) =", r_squared_test)

# Slice only PSa part
true_values = true_values[:, len(input_cols):]
mean_array = mean_array[:, len(input_cols):]

# R² for PSa only
r_squared_test = r2_score(true_values, mean_array)
print("Final R² on test set (PSa only):", r_squared_test)

# Scatter plot
plt.figure(figsize=(6, 6))
plt.scatter(true_values, mean_array, color='royalblue', s=10, label='Ideal Fit (y = x)')
min_val = min(np.min(true_values), np.min(mean_array))
max_val = max(np.max(true_values), np.max(mean_array))
plt.plot([min_val, max_val], [min_val, max_val], 'k--', label=f'R² = {r_squared_test:.2f}')
plt.title('R² Plot (Test Set)')
plt.xlabel('Recorded PSa (log₁₀)')
plt.ylabel('Predicted PSa (log₁₀)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

generator.load_state_dict(torch.load('best_generator.pth'))
discriminator.load_state_dict(torch.load('best_discriminator.pth'))

conditions_test = X_test_scaled[input_cols].to_numpy()
true_values = X_test_scaled[input_cols.tolist() + target_cols.tolist()].to_numpy()

outputs = []
for _ in range(1000):
    out = predict(conditions_test, generator, latent_dim=latent_dim, seed = _) # Pass latent_dim
    outputs.append(out)

mean_output = np.mean(outputs, axis=0)

r_squared_test = r2_score(true_values[:, len(input_cols):], mean_output[:, len(input_cols):])
print("Final R² on test set (PSa only):", r_squared_test)

import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style
sns.set(style='whitegrid', context='notebook', font_scale=1.2)

# Create the figure
plt.figure(figsize=(7, 7))

# Slice only PSa part for plotting
true_values_psa = true_values[:, len(input_cols):]
mean_array_psa = mean_output[:, len(input_cols):] # Use mean_output instead of mean_array

plt.scatter(true_values_psa, mean_array_psa, color='steelblue', edgecolor='black', alpha=0.6, s=25)

# Ideal fit line
min_val = min(np.min(true_values_psa), np.min(mean_array_psa))
max_val = max(np.max(true_values_psa), np.max(mean_array_psa))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label=f'Ideal Fit (y = x)\nR² = {r_squared_test:.2f}')

# Add title and labels with formatting
plt.title('Predicted vs Recorded Spectral Acceleration (log₁₀)', fontsize=14, weight='bold')
plt.xlabel('Recorded PSa (log₁₀)', fontsize=12)
plt.ylabel('Predicted PSa (log₁₀)', fontsize=12)

# Ticks and limits
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)

# Legend and grid
plt.legend(loc='upper left', fontsize=11)
plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)

# Tight layout and show
plt.tight_layout()
plt.show()