import matplotlib.pyplot as plt 
import numpy as np

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.8, 9.6))

epochs = list(range(1, 61))
total_train_losses = np.power(epochs, 2)
total_valid_losses = np.power(epochs, 3)

kl_train_losses = np.power(epochs, 1)
kl_valid_losses = np.power(epochs, 1)

train_reconstruction_losses = np.power(epochs, 2)
valid_reconstruction_losses = np.power(epochs, 3)

# Plot combined loss 
ax1.plot(epochs, total_train_losses, 'b', label='Train')
ax1.plot(epochs, total_valid_losses, 'r', label='Valid')
ax1.set(xlabel="Epochs", ylabel="Loss")
ax1.legend(prop={'size': 10})
ax1.set_title("Combined Loss")
ax1.set_yscale('log')

# Plot KL 
ax2.plot(epochs, kl_train_losses, 'b', label='Train')
ax2.plot(epochs, kl_valid_losses, 'r', label='Valid')
ax2.set(xlabel="Epochs", ylabel="Loss")
ax2.legend(prop={'size': 10})
ax2.set_title("KL Loss")

# Plot reconstruction loss 
ax3.plot(epochs, train_reconstruction_losses, 'b', label='Train')
ax3.plot(epochs, valid_reconstruction_losses, 'r', label='Valid')
ax3.set(xlabel="Epochs", ylabel="Loss")
ax3.legend(prop={'size': 10})
ax3.set_title("Reconstruction Loss")
ax3.set_yscale('log')

plt.tight_layout()

plt.savefig('./model_graphs/model_losses_1.png')