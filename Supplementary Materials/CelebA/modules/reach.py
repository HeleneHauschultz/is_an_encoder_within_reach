import io
import random
from itertools import islice

import matplotlib.pyplot as plt
import numpy as np
import torch as t
import tqdm
from PIL import Image
from einops import rearrange
from sklearn.neighbors import BallTree
from torch.utils.data import DataLoader, Dataset

from tasks.roll.roll import swiss_roll_decoder


def batched_cdist(x_out, x0, batch_size):
    batches = np.array_split(np.arange(len(x_out)), batch_size)
    dists = []
    for batch in batches:
        dists.append(t.cdist(x_out[batch].to(x0.device), x0).squeeze().cpu())
    return t.cat(dists)


def sample_hyperball(x0, radius, n):
    device = x0.device
    d = len(x0)
    u = t.randn(n, d, device=device)  # an array of d normally distributed random variables
    norm = t.sqrt(t.sum(u ** 2, dim=1, keepdim=True))
    r = radius * t.rand(n, 1, device=device) ** (1.0 / d)
    nball_samples = r * u / norm
    return x0 + nball_samples


def reach_est_point_cloud(dataset: Dataset, encoder, decoder, batch_size=1024):
    t.set_grad_enabled(False)

    print("Computing reach...")
    reaches = []
    errors = []
    for i in tqdm.tqdm(range(len(dataset))):
        x_in = dataset[i][None]
        x0, jac = decoder(encoder(x_in), jacobian=True)
        errors.append(t.sqrt(((x0 - x_in.view(1, -1).to(x0.device)) ** 2).sum()))
        reach = t.inf
        radius = 0.01
        reach_iters = []
        pbar = tqdm.tqdm(range(100))
        for _ in pbar:
            cloud_in = sample_hyperball(x0[0], radius, batch_size)
            cloud = decoder(encoder(rearrange(cloud_in, "b (c h w) -> b c h w", b=batch_size, c=3, h=64, w=64)))
            reach = min(reach, reach_est(x0[0], jac[0], cloud).item())
            radius = 2 * reach
            pbar.set_description("r: %.3f" % reach)
            reach_iters.append(reach)

        reaches.append(reach_iters)
        t.save({'reach': reaches, 'error': errors}, 'reach.t')

    return t.tensor(reaches), t.tensor(errors)


def reach_est_simple(dataset: Dataset, encoder, decoder, batch_size=1024):
    """
    :param dataset:
    :param encoder: (b, ...) -> (b, z)
    :param decoder: (b, z) -> (b, x), (b, x, z)
    :param batch_size: int
    :return:
    """
    t.set_grad_enabled(False)
    print("Encoding/Decoding dataset...")
    dl = DataLoader(dataset, batch_size=batch_size)

    x_out = []
    errors = []
    for batch in tqdm.tqdm(dl):
        decoded = decoder(encoder(batch)).cpu()
        x_out.append(decoded)
        errors.append(t.sqrt(((decoded - batch.view(batch.shape[0], -1)) ** 2).sum(dim=1)))
    x_out = t.cat(x_out, dim=0)
    errors = t.cat(errors, dim=0)

    print("Creating BallTree...")
    bt = BallTree(x_out.numpy())

    print("Computing reach...")
    n_samples = len(dataset)
    reaches = []
    pbar = tqdm.tqdm(range(n_samples))
    for i in pbar:
        x0, jac = decoder(encoder(dataset[i][None]), jacobian=True)
        checked = {i}
        reach = errors[i]
        # start = time.time()
        # distances = batched_cdist(x_out, x0, batch_size)
        # took = time.time() - start
        # all = t.arange(n_samples)
        while True:
            samples_in_range = set(bt.query_radius(x0.cpu().numpy(), 2 * reach)[0])  # set(all[distances < 2 * reach].numpy())
            to_check = samples_in_range - checked
            if len(to_check) == 0:  # no samples left to check
                break
            else:
                batch = list(islice(to_check, batch_size))
                cloud = x_out[batch].to(x0.device)
                reach = min(reach, reach_est(x0[0], jac[0], cloud).item())
                checked = checked.union(batch)
        # pbar.set_description("error: %.3f, reach: %.3f, checked/total: %d/%d" % (errors[i], reach, len(checked), n_samples))
        reaches.append(reach)
    return t.tensor(reaches), errors


def reach_est(x0, jac, cloud):
    """
    :param x0: (x)
    :param jac: (x, z)
    :param cloud: (b, x)
    :return:
    """
    Px0 = t.eye(len(x0), device=x0.device) - t.linalg.lstsq(jac.T, jac.T).solution
    r_min = t.min((t.linalg.norm(cloud - x0, dim=1) ** 2 / (2 * t.linalg.norm(Px0 @ (cloud - x0).T, dim=0))))

    return r_min


def reach_sample_est(dataset, encoder, decoder, batch_size=128, max_iters=1_000_000, n_iters_no_diff_max=100):
    """
    :param dataset:
    :param encoder: (b, ...) -> (b, z)
    :param decoder: (b, z) -> (b, x), (b, x, z)
    :param batch_size: int
    :param max_iters: int
    :param n_iters_no_diff_max: int
    :return:
    """
    t.set_grad_enabled(False)
    n_samples = len(dataset)
    reach = float("inf") * t.ones(n_samples, device="cpu")

    pbar = tqdm.tqdm(range(max_iters))
    n_iters_no_diff = 0
    diffs = []
    for _ in pbar:
        idx = random.sample(range(n_samples), batch_size)
        batch = t.stack([dataset[i] for i in idx])
        z = encoder(batch)
        _, plr = pointwise_local_reach_est(z, decoder)
        new_reach = t.minimum(reach[idx], plr.cpu())
        diff = (reach[idx] - new_reach).mean()
        reach[idx] = new_reach

        pbar.set_description("diff: %.3f, n_iters_no_diff: %d" % (diff, n_iters_no_diff))
        diffs.append(diff)

        n_iters_no_diff = 0 if diff > 0 else n_iters_no_diff + 1
        if n_iters_no_diff > n_iters_no_diff_max:
            break

    return reach, diffs


def pointwise_local_reach_est(z: t.Tensor, decoder):
    """
    :param z: (N, Z)
    :param decoder: (N, Z) -> (N, X), (N, X, Z)
    :return: decoded points and reach (N,X), (N, )
    """

    x, jac = decoder(z)

    diff = x.unsqueeze(0) - x.unsqueeze(1)  # (N, N, X)
    dim = x.shape[1]
    pjac = t.permute(jac, (0, 2, 1))
    proj = t.eye(dim, device=z.device) - jac @ t.linalg.solve(pjac @ jac, pjac)
    dnorm_squared = t.linalg.norm(diff, dim=2) ** 2
    proj_norm = (2 * t.linalg.norm((proj @ t.permute(diff, (0, 2, 1))), dim=1))
    R = dnorm_squared / (proj_norm + 1e-16)
    R.fill_diagonal_(1e16)
    r, _ = R.min(dim=1)
    assert t.all(t.isfinite(r)), "non-finite reach"

    return x, r


def plot_reach(x, reach, manifold=None, x_train=None, return_im=False):
    plt.figure()
    _, ax = plt.subplots(dpi=150)
    ax.scatter(x[:, 0], x[:, 1])

    if x_train is not None:
        plt.plot(x_train[:, 0], x_train[:, 1], 'g.', label='train')

    if manifold is not None:
        plt.plot(manifold[:, 0], manifold[:, 1], 'r-', label='manifold')

    for _xy, _r in zip(x, reach):
        circle = plt.Circle(xy=(_xy[0], _xy[1]), radius=_r, fill=False, alpha=0.2)
        ax.add_patch(circle)

    plt.axis("equal")
    plt.legend()
    if return_im:
        with io.BytesIO() as buf:
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            return np.array(Image.open(buf))[:, :, :3]
    else:
        plt.show()


if __name__ == '__main__':
    z = t.linspace(2 * t.pi, 6 * t.pi, 1000)[:, None]
    manifold = swiss_roll_decoder(z)
    z_samples = 2 * t.pi + 4 * t.pi * t.rand(200, 1)
    x = swiss_roll_decoder(z_samples)

    reach, diffs = reach_sample_est(z_samples, swiss_roll_decoder, batch_size=128)

    plot_reach(x, reach, manifold)
