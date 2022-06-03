import torch as t
import tqdm
from torch.distributions import Normal
from torch.nn.functional import softplus
from torch.nn.utils import clip_grad_norm_
from pathlib import Path

from reach import pointwise_local_reach_est, l2_reach_est


from stochman import nnj


class AutoEncoder(t.nn.Module):

    def __init__(self, alpha, encoder, decoder,  lr=1e-3, beta = 0, output_dir = 'out', l2_reg = False):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.encoder = encoder
        self.decoder = decoder
        self.output_dir = output_dir
        self.device = "cuda" if t.cuda.is_available() else "cpu"
        self.optim = t.optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)

        # check if output dir exists
        if not Path(self.output_dir).exists():
            Path(self.output_dir).mkdir(parents=True)
        assert Path(self.output_dir).is_dir(), 'Warning. self.output_dir is NOT a directory.'

        #print(self)
        #for n, p in self.named_parameters():
        #    print(n, p.shape, p.dtype)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def decoder_jac(self, z):
        _, D = self.decoder(z,jacobian = True)
        return D
        # return t.stack([t.autograd.functional.jacobian(self.decoder, zz, create_graph=True) for zz in z])

    def do_train(self, x_train, n_train_iter=20_000):
        x_train = x_train.to(self.device)
        pbar = tqdm.tqdm(range(n_train_iter))
        losses = []
        l2_losses = []
        reach_losses = []
        for _ in pbar:
            z = self.encoder(x_train)
            z_space = t.linspace(z.min().item(), z.max().item(), 1000, device=self.device)[:, None]
            zz = t.cat([z, z_space], dim=0)

            x_dec, reach = pointwise_local_reach_est(zz, self.decoder)
            #x_dec, reach = pointwise_cloud_reach_est(z,reach,self.encoder,self.decoder,self.decoder_jac)
            x_dec = x_dec[:x_train.shape[0]]
            reach = reach[:x_train.shape[0]]
            error = ((x_dec - x_train) ** 2).sum(dim=1).sqrt()
            reach_loss = softplus(error - reach).mean()
            l2_reg = 0
            if l2_reg == True:
                l2_reg = sum(p.pow(2.0).sum() for p in self.parameters())
            l2_loss = ((x_dec - x_train) ** 2).mean()
            loss = l2_loss + self.alpha * reach_loss + self.beta*l2_reg
            assert t.isfinite(loss), "non-finite loss"
            self.optim.zero_grad()
            loss.backward()
            for p in self.parameters():
                assert t.all(t.isfinite(p.grad)), "non-finite gradients"

            clip_grad_norm_(self.parameters(), 10.0)
            self.optim.step()
            losses.append(loss.item())
            l2_losses.append(l2_loss.item())
            reach_losses.append(reach_loss.item())
            pbar.set_description("l2 loss: %.3f, reg loss: %.3f" % (l2_loss.item(), reach_loss.item()))
            #if self.batch_idx % self.report_freq == 0:
            #    z = self.encoder(x_train)
            #    x_dec, reach = pointwise_local_reach_est(z, self.decoder, self.decoder_jac)
            #    self.report(loss.item(), l2_loss.item(), reach_loss.item(), z, x_dec, reach, x_train)
            #self.batch_idx += 1
        t.save(losses, f"{self.output_dir}/losses.pt")
        t.save(l2_losses, f"{self.output_dir}/l2_loss.pt")
        t.save(reach_losses, f"{self.output_dir}/reach_losses.pt")

    def save(self, fn, overwrite=False):
        file_path = f"{self.output_dir}/{fn}"
        if not Path(file_path).exists() or overwrite is True:
            t.save({
                #'batch_idx': self.batch_idx,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optim.state_dict(),
            }, file_path)
        else:
            print("Warning. The file ({file_path}) already exists - the model will not be saved.")

    def load(self, fn):
        checkpoint = t.load(fn, map_location=t.device(self.device))
        #self.batch_idx = checkpoint["batch_idx"]
        self.load_state_dict(checkpoint["model_state_dict"])
        self.optim.load_state_dict(checkpoint["optimizer_state_dict"])


