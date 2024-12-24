import numpy as np
import SpinModel_fm as smk
from matplotlib.pyplot import subplots as spp

x = np.linspace(1.34, 7, 10)
y = np.linspace(0.5, 5, 15)
x, y = np.meshgrid(x, y)
ny, nx = np.shape(x)

z = np.zeros((ny, nx))

i = 0
while i < ny:
    j = 0
    while j < nx:
        cp_strg = [-3, -2.5, 2.5, 0.1, x[i, j]]
        Hlsw = smk.fm_spin_model(cp_strg)
        kappa, N = smk.unnormalized_heat_conductivity(Hlsw, 1, 0, y[i, j], 10)
        z[i, j] = z[i, j] + np.sum(kappa)
        j += 1
    i += 1

fig, ax = spp(facecolor='0.95')
pcm = ax.contourf(x, y, z)

ax.set_title("$\kappa^{yx}$ change w.r.t. external field $B$ and temperature $T$")
ax.set_xlabel("h (meV)")
ax.set_ylabel("T(K)$")

fig.colorbar(pcm, ax=ax)
fig.savefig('kappa_vs_BT.png')