import ML.agent as a
import torch
import matplotlib.pyplot as plt
import numpy as np

filename = "model.pt"

model = a.Model(8)
model.load_state_dict(torch.load(filename))

sd = model.state_dict()
conv1_weight = sd['conv1.weight']

dim1 = 8
dim2 = 8

min_val = torch.min(conv1_weight).item()
max_val = torch.max(conv1_weight).item()

plt.figure(figsize=(10,10))

for r in range(dim1):
    for c in range(dim2):
        i = dim1 * r + c
        conv1_weight_np = conv1_weight[i,0].numpy()
        #min_val = np.min(conv1_weight_np)
        #max_val = np.max(conv1_weight_np)

        conv1_weight_np = (conv1_weight_np - min_val) / (max_val - min_val) * 255
        plt.subplot(dim1, dim2, i + 1)
        plt.imshow(conv1_weight_np, cmap='Blues')
        
plt.show()