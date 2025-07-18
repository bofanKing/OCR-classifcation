# plot_results.py

import numpy as np
import matplotlib.pyplot as plt

history = np.load('models/history.npy')

plt.figure()
plt.plot(history[:, 0], label='Train Loss')
plt.plot(history[:, 1], label='Val Loss')
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_curve.png")
plt.show()

plt.figure()
plt.plot(history[:, 2], label='Train Acc')
plt.plot(history[:, 3], label='Val Acc')
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("acc_curve.png")
plt.show()
