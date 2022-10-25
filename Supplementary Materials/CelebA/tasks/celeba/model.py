import os
from typing import Tuple

import torch as t
from einops import rearrange
from stochman import nnj
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets

from modules.iterable_dataset_wrapper import IterableWrapper
from modules.model import Model
from modules.reach import reach_est_simple, reach_est_point_cloud
from util import get_writers


class CelebA(Model):
    def __init__(self):
        super().__init__()
        batch_size = 128
        filter_size = 5
        pad = filter_size // 2
        encoder_hid = 128
        h = w = 64
        n_channels = 3

        self.device = "cuda" if t.cuda.is_available() else "cpu"

        data_dir = os.environ.get('DATA_DIR') or "data"
        tp = transforms.Compose([transforms.Resize((h, w)), transforms.ToTensor()])
        self.train_set, self.val_set, self.test_set = [datasets.CelebA(data_dir, split=split, download=True, transform=tp) for split in ["train", "valid", "test"]]
        self.train_loader = iter(DataLoader(IterableWrapper(self.train_set), batch_size=batch_size, pin_memory=True))
        self.val_loader = iter(DataLoader(IterableWrapper(self.val_set), batch_size=batch_size, pin_memory=True))
        self.train_writer, self.test_writer = get_writers("reach-celeba")

        self.encoder = nn.Sequential(  # (bs, 3, 64, 64)
            nn.Conv2d(n_channels, encoder_hid, filter_size, padding=pad), nn.ELU(),  # (bs, hid, 64, 64)
            nn.Conv2d(encoder_hid, encoder_hid, filter_size, padding=pad, stride=2), nn.ELU(),  # (bs, hid, 32, 32)
            nn.Conv2d(encoder_hid, encoder_hid, filter_size, padding=pad, stride=2), nn.ELU(),  # (bs, hid, 16, 16)
            nn.Conv2d(encoder_hid, encoder_hid, filter_size, padding=pad, stride=2), nn.ELU(),  # (bs, hid, 8, 8)
            nn.Conv2d(encoder_hid, encoder_hid, filter_size, padding=pad, stride=2), nn.ELU(),  # (bs, hid, 4, 4),
            nn.Conv2d(encoder_hid, encoder_hid, filter_size, padding=pad, stride=2), nn.ELU(),  # (bs, hid, 2, 2),
            nn.Conv2d(encoder_hid, encoder_hid, filter_size, padding=pad, stride=2), nn.ELU(),  # (bs, hid, 1, 1),
        )

        self.decoder = nnj.Sequential(  # (bs, hid, 1, 1),
            nnj.ConvTranspose2d(encoder_hid, encoder_hid, filter_size, stride=2, padding=pad, output_padding=1), nnj.ELU(),  # (bs, hid, 2, 2),
            nnj.ConvTranspose2d(encoder_hid, encoder_hid, filter_size, stride=2, padding=pad, output_padding=1), nnj.ELU(),  # (bs, hid, 4, 4),
            nnj.ConvTranspose2d(encoder_hid, encoder_hid, filter_size, stride=2, padding=pad, output_padding=1), nnj.ELU(),  # (bs, hid, 8, 8),
            nnj.ConvTranspose2d(encoder_hid, encoder_hid, filter_size, stride=2, padding=pad, output_padding=1), nnj.ELU(),  # (bs, hid, 16, 16),
            nnj.ConvTranspose2d(encoder_hid, encoder_hid, filter_size, stride=2, padding=pad, output_padding=1), nnj.ELU(),  # (bs, hid, 32, 32),
            nnj.ConvTranspose2d(encoder_hid, encoder_hid, filter_size, stride=2, padding=pad, output_padding=1), nnj.ELU(),  # (bs, hid, 64, 64),
            nnj.ConvTranspose2d(encoder_hid, n_channels, filter_size, padding=pad),  # (bs, 3, 64, 64),
        )
        self.decoder[-1].bias.data = next(self.train_loader)[0].mean(dim=(0, 2, 3))

        print(self)
        for n, p in self.named_parameters():
            print(n, p.shape)

        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)
        self.batch_idx = 0

    def compute_reach(self):
        def encoder(x):
            x = x.to(self.device)
            z: t.Tensor = self.encoder(x)  # (b,c,h,w)
            return rearrange(z, "b c h w -> b (c h w)")

        def decoder(z, jacobian=False):
            z = rearrange(z, "b (c h w) -> b c h w", c=128, h=1, w=1)  # (b,c,h,w)
            if not jacobian:
                x = self.decoder(z)
                return rearrange(x, "b c h w -> b (c h w)")
            else:
                x, jac = self.decoder(z, jacobian=True)  # (1, 3, 64, 64), (1, 3, 64, 64, 128, 1, 1)
                return (
                    rearrange(x, "b c h w -> b (c h w)"),
                    rearrange(jac, "b xc xh xw zc zh zw -> b (xc xh xw) (zc zh zw)")
                )

        class InputOnlyDataset(Dataset):
            def __init__(self, delegate):
                self.delegate = delegate

            def __getitem__(self, item):
                return self.delegate[item][0]

            def __len__(self):
                return len(self.delegate)

        train_x_only = InputOnlyDataset(self.val_set)

        reach_est_point_cloud(train_x_only, encoder, decoder)

    def forward(self, x) -> Tuple[t.Tensor, t.Tensor]:
        x = x.to(self.device)
        z = self.encoder(x)
        decoded = self.decoder(z)
        loss = ((x - decoded) ** 2).mean()
        return decoded, loss

    def train_batch(self) -> float:
        self.train(True)

        x, y = next(self.train_loader)
        decoded, loss = self.forward(x)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # t.nn.utils.clip_grad_norm_(self.parameters(), 1.0, error_if_nonfinite=True)

        if self.batch_idx % 1000 == 0:
            self.report(self.train_writer, loss, x, decoded)

        self.batch_idx += 1
        return loss.item()

    def eval_batch(self) -> float:
        self.train(False)
        with t.no_grad():
            x, y = next(self.val_loader)
            decoded, loss = self.forward(x)
            self.report(self.test_writer, loss, x, decoded)
        return loss.item()

    def report(self, writer, loss, x, decoded):
        writer.add_scalar("loss/loss", loss, self.batch_idx)
        writer.add_images("x", t.clamp(x[:64], 0, 1), self.batch_idx)
        writer.add_images("decoded", t.clamp(decoded[:64], 0, 1), self.batch_idx)

    def test(self, n_iw_samples):
        pass  # todo


if __name__ == '__main__':
    m = CelebA()
    m.eval_batch()
    m.train_batch()
