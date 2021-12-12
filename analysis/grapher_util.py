"""
    grapher_util.py

    Given the log data tensor from logger, creates the loss and accuracy graphs.

"""


import torch
import numpy as np
from sys import argv, exit
import os
from matplotlib import pyplot as plt

error_msg = "Usage: python /path/to/grapher_util.py <PATH/TO/LOGDATA/TENSOR>"
if len(argv) != 2:
    exit(error_msg)

path_to_data = argv[1]
data = torch.load(path_to_data)
batch_per_epoch = data.shape[1]

# finds the total numbers of epoch to be graphed
# (assumes that the final epoch to be included in the graph was greater than 0)
total_e = 0
is_nonzero = True

while is_nonzero:
    # any batch in that epoch should have non-zero training loss
    # since we save only once every epoch
    total_e += 1
    is_nonzero = torch.is_nonzero(data[total_e, 0, 0])

epoch_arr = np.arange(start=0, stop=total_e)

train_loss_arr = np.ndarray((len(epoch_arr)))
train_acc_arr = np.ndarray((len(epoch_arr)))
test_loss_arr = np.ndarray((len(epoch_arr)))
test_acc_arr = np.ndarray((len(epoch_arr)))

for e in range(total_e):

    train_loss_arr[e] = data[e, :, 0].mean()
    train_acc_arr[e] = data[e, :, 1].mean()
    test_loss_arr[e] = data[e, :, 2].mean()
    test_acc_arr[e] = data[e, :, 3].mean()

if not os.path.exists('graphs'):
    os.makedirs('graphs')


plt.plot(epoch_arr, train_loss_arr, color="red", label="Training Loss")
plt.plot(epoch_arr, test_loss_arr, color="blue", label="Testing Loss")
plt.title("Cross-Entropy Loss over Training Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("analysis/graphs/loss.png")
plt.close()

plt.plot(epoch_arr, train_acc_arr, color="red", label="Training Accuracy")
plt.plot(epoch_arr, test_acc_arr, color="blue", label="Testing Accuracy")
plt.title("Classification Accuracy over Training Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("analysis/graphs/acc.png")
plt.close()
