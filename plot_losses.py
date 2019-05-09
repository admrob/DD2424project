#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 10:07:33 2019

@author: delavergne
"""

import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

log_data = pd.read_csv('training_vae.log', sep=',', engine='python')

plt.plot(log_data.epoch, log_data.kl_loss, 'b')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.title('KL loss')
plt.show()

plt.plot(log_data.epoch, log_data.xent_loss, 'r')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.title('Reconstruction loss')
plt.show()

plt.plot(log_data.epoch, log_data.loss, 'g')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.title('Combined loss')
plt.show()