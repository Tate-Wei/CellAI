
import numpy as np
import matplotlib.pyplot as plt

data = np.load('/home/mq/Documents/AMI/Group02/web/src/model/inputs/a_learning/temp/pr_dataset.npy', allow_pickle=True).item()
img = data[11]['ori_ph']

num_images = 10
num_rows = 2
num_cols = 5
num =1


fig, axes = plt.subplots(num_rows, num_cols)
for i,ax in enumerate(axes.ravel()):
    if i >= num_images:
        ax.axis('off')
        break
    ax.imshow(data[i+num*10]['ph'])
    ax.set_title(f"Predicted: {data[i+num*10]['label']}, Entropy: {data[i+num*10]['conf']:.3f}")
    ax.axis('off')

plt.tight_layout()
plt.show()