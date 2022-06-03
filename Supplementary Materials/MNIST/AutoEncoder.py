import torch
from tqdm import tqdm
from torch.distributions import Normal
from torch.nn.functional import softplus
from torch.nn.utils import clip_grad_norm_

from pathlib import Path 


from stochman import nnj

class AutoEncoder(torch.nn.Module):

	def __init__(self, encoder, decoder, lr =1e-3, output_dir = 'out'):
		super().__init__()

		self.encoder = encoder
		self.decoder = decoder
		self.optim = torch.optim.Adam(self.parameters(), lr = lr)
		self.output_dir = output_dir
		
		# check if output dir exists
		if not Path(self.output_dir).exists():
			Path(self.output_dir).mkdir(parents=True)
		assert Path(self.output_dir).is_dir(), 'Warning. self.output_dir is NOT a directory.'


	def forward(self,x):
		return self.decoder(self.encoder(x))

	def do_train(self, x_train, n_train_iter = 10_000, regularization = 'none'):
		pbar = tqdm(range(n_train_iter))

		if regularization == 'none':
			losses = []
			for _ in pbar:
				x_dec = self.decoder(self.encoder(x_train))
				loss = ((x_dec-x_train)**2).mean()
				assert torch.isfinite(loss), "non-finite loss"
				self.optim.zero_grad()
				loss.backward()
				for p in self.parameters():
					assert torch.all(torch.isfinite(p.grad)), "non-finite gradients"

				clip_grad_norm_(self.parameters(), 10.0)
				self.optim.step()
				losses.append(loss.item())
				pbar.set_description(f"l2 loss: {loss.item()}")
			torch.save(losses, f"{self.output_dir}/loss_curve.t")
	




	def save(self, fn, overwrite=False):
		file_path = f"{self.output_dir}/{fn}"
		if not Path(file_path).exists() or overwrite is True:
			torch.save({
				#'batch_idx': self.batch_idx,
				'model_state_dict': self.state_dict(),
				'optimizer_state_dict': self.optim.state_dict(),
			}, file_path)
		else:
			print("Warning. The file ({file_path}) already exists - the model will not be saved.")
	
	def load(self, fn):
		checkpoint = torch.load(fn)
		#self.batch_idx = checkpoint["batch_idx"]
		self.load_state_dict(checkpoint["model_state_dict"])
		self.optim.load_state_dict(checkpoint["optimizer_state_dict"])		