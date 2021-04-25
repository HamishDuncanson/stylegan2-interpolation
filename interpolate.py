import numpy as np
import torch
from stylegan2 import models

jump = 0.01
trunc = 1
static = True
stop = 400
size = 256
scale = 0.81

frames = []
G = models.load("Gs.pth")
G.eval()
if static:
    G.static_noise()
G.set_truncation(trunc)
dim = G.latent_size
prv = np.random.randn(dim) / scale
pv_0 = prv
for _ in range(10000):
    c_r = np.random.randn(dim) / scale
    steps = np.ceil(np.sqrt(np.square(c_r - prv).mean()) / jump).astype(int)
    n_w = np.linspace(prv, c_r, steps)[1:]
    with torch.no_grad():
        for x in n_w:
            im = G(torch.from_numpy(x.reshape(1, -1)).to(dtype=torch.float32))[0]
            frames += [np.clip((im.numpy().transpose(1,2,0)+1)*128,0,255)]
    prv = c_r
    if len(frames) > stop:
        c_r = pv_0
        steps = np.ceil(np.sqrt(np.square(c_r - prv).mean()) / jump).astype(int)
        n_w = np.linspace(prv, c_r, steps)[1:]
        with torch.no_grad():
            for x in n_w:
                im = G(torch.from_numpy(x.reshape(1,-1)).to(dtype=torch.float32))[0]
                frames += [np.clip((im.numpy().transpose(1,2,0)+1)*128,0,255)]
        break
np.save("interpolation", np.stack(frames))




