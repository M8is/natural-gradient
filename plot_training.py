import torch
from matplotlib import pyplot as plt

import models
import quanser_robots

model = torch.load("nac_lstd_model.pt")

plt.ion()
plt.show()

f, (ax1) = plt.subplots(1, 1)

ax1.set_title('Total Returns')
ax1.plot(model.returns)

plt.draw()
plt.pause(0.05)

while True: pass
