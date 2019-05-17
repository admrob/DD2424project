
import pandas as pd
import matplotlib.pyplot as plt
import sys

plt.style.use('ggplot')

log_data = pd.read_csv('training_vae_mnist_no_optimization.log', sep=',', engine='python')
log_data2 = pd.read_csv('training_vae_mnist_with_optimization.log', sep=',', engine='python')

plt.plot(log_data.epoch, log_data.kl_loss, 'b', label='No optimization')
plt.plot(log_data2.epoch, log_data2.kl_loss, 'r', label='With optimization')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.title('KL loss')
plt.legend()
plt.show()

plt.plot(log_data.epoch, log_data.xent_loss, 'b', label='No optimization')
plt.plot(log_data2.epoch, log_data2.xent_loss, 'r', label='With optimization')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.title('Reconstruction loss')
plt.legend()
plt.show()

plt.plot(log_data.epoch, log_data.loss, 'b', label='No optimization')
plt.plot(log_data2.epoch, log_data2.loss, 'r', label='With optimization')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.title('Combined loss')
plt.legend()
plt.show()