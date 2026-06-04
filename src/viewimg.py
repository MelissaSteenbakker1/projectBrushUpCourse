import matplotlib.pyplot as plt
from importdata import load_chestmnist_dataset

dataset = load_chestmnist_dataset("train")

image, label = dataset[500]  # change index to inspect different samples

plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Label: {label}")
plt.savefig("images/sample.png") # Open saved image file in ./images/sample.png