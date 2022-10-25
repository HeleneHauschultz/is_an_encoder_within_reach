import numpy as np
import torch as t
import matplotlib.pyplot as plt

sd = t.load('small-r.t', map_location=t.device('cpu'))
sr = t.tensor(sd["reach"])
se = t.tensor(sd["error"])

ld = t.load('large-r.t', map_location=t.device('cpu'))
lr = t.tensor(ld["reach"])
le = t.tensor(ld["error"])

n_min = min(len(le), len(se))
assert t.allclose(se[:n_min], le[:n_min])

plt.figure(dpi=300)
# plt.title("Sensitivity analysis of initial search radius")
plt.plot(sr[:n_min, -1], sr[:n_min, -1], 'r-')
plt.plot(lr[:n_min, -1], sr[:n_min, -1], 'b.', alpha=0.1)
plt.xlabel("Reach ($r_0 = 1.0$)")
plt.ylabel('Reach ($r_0 = 0.01$)')
plt.savefig('r0-sens-analysis.pdf')

plt.figure(dpi=300)
# plt.title("Convergence analysis")
n_reach = (lr / lr[:, :1]).numpy().T  # (100, n)
pps = np.percentile(n_reach, q=np.arange(1, 100), axis=1).T  # (100, 99)
plt.plot(np.arange(pps.shape[0]), pps[:, 49], 'b')
for p in range(49):
    plt.fill_between(np.arange(pps.shape[0]), pps[:, p], pps[:, -(p + 1)], color='b', alpha=0.02, lw=0)
plt.ylabel("Normalized reach")
plt.xlabel('Batches')
plt.savefig('norm-reach-v-batches.pdf')

plt.figure(dpi=300)
# plt.title("Reach vs. Error")
plt.plot(le, le, 'r-')
plt.plot(le, lr[:, -1], 'b.', alpha=0.1)
plt.ylabel("Reach")
plt.xlabel('Error')
plt.savefig('reach-v-error.pdf')
