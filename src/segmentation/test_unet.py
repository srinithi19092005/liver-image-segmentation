import torch

from unet import UNet


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 60)
print("U-NET TEST")
print("=" * 60)

print("\nDevice:")
print(device)


# ============================================================
# CREATE MODEL
# ============================================================

model = UNet().to(device)


print("\nModel created successfully.")


# ============================================================
# TEST INPUT
# ============================================================

x = torch.randn(
    2,
    1,
    256,
    256
).to(device)


print("\nInput shape:")
print(x.shape)


# ============================================================
# FORWARD PASS
# ============================================================

with torch.no_grad():

    liver_output, tumor_output = model(x)


# ============================================================
# OUTPUTS
# ============================================================

print("\nLiver output shape:")
print(
    liver_output.shape
)


print("\nTumor output shape:")
print(
    tumor_output.shape
)


# ============================================================
# CHECK
# ============================================================

assert liver_output.shape == (
    2,
    1,
    256,
    256
)

assert tumor_output.shape == (
    2,
    1,
    256,
    256
)


print("\n" + "=" * 60)
print("U-NET TEST SUCCESSFUL")
print("=" * 60)