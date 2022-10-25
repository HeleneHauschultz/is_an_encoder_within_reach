import torch
from stochman import nnj
import tqdm

def pointwise_local_reach_est(z: torch.Tensor, decoder):
	x, jac = decoder(z, jacobian = True)

	diff = x.unsqueeze(0)-x.unsqueeze(1)
	dim = x.shape[1]
	pjac = torch.permute(jac,(0,2,1))
	proj = torch.eye(dim)-jac@torch.linalg.solve(pjac@jac,pjac)

	d_norm_squared = torch.linalg.norm(diff, dim = 2)**2
	proj_norm = (2*torch.linalg.norm((proj@torch.permute(diff,(0,2,1))), dim=1))
	R = d_norm_squared / (proj_norm + 1e-16)
	R.fill_diagonal_(1e16)

	r, _ = R.min(dim=1)
	assert torch.all(torch.isfinite(r)), "non-finite reach"

	return x, r 

def l2_reach_est(x_in: torch.Tensor, decoder, encoder, cloud_size = 1000):
	z: torch.Tensor = encoder(x_in)
	x_out, jac = decoder(z, jacobian = True)
	pjac = torch.permute(jac, (0,2,1))
	dim = x_out.shape[1]
	proj = torch.eye(dim)-jac @ torch.linalg.solve(pjac @ jac, pjac)

	l2_error = torch.norm(x_in - x_out, dim = 1)

	r = torch.zeros(len(x_in))

	for i in (range(len(x_in))):
		cloud = uniform_cloud(2*l2_error[i], cloud_size) + x_out[i]
		proj_cloud = decoder(encoder(cloud))-x_out[i]
		r[i] = torch.min(torch.linalg.norm(proj_cloud, dim =1)**2/(2*torch.linalg.norm(proj[i]@proj_cloud.T, dim = 0)))

	return x_out,r

def uniform_cloud(radius, n_samples):
	u = torch.randn(n_samples,784)
	norm = torch.norm(u, dim = 1)
	r = radius*torch.rand(n_samples)**(1.0/784)
	x = (r/norm).unsqueeze(1)*u
	return x