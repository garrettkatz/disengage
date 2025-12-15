import numpy as np
import matplotlib.pyplot as pt

H = W = 4096
alpha = .5
img = alpha * np.random.rand(H, W, 3) + (1 - alpha) * np.array([246/255, 215/255, 176/255])
pt.figure(figsize=(10,10))
pt.imshow(img)
pt.savefig("noisy_sand.png")
pt.show()

