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

def l2_reach_est(x_in: torch.Tensor, decoder, encoder, cloud_size = 100):
	z: torch.Tensor = encoder(x_in)
	x_out, jac = decoder(z, jacobian = True)
	pjac = torch.permute(jac, (0,2,1))
	dim = x_out.shape[1]
	proj = torch.eye(dim)-jac @ torch.linalg.solve(pjac @ jac, pjac)

	l2_error = torch.norm(x_in - x_out, dim = 1)

	r = reach_est(x_out, proj, l2_error, encoder, decoder, cloud_size)

	

	return x_out,r

def cloud_reach_est(x_in, decoder, encoder, radius = 10,cloud_size = 100):
	z: torch.Tensor = encoder(x_in)
	x_out, jac = decoder(z, jacobian = True)
	pjac = torch.permute(jac, (0,2,1))
	dim = x_out.shape[1]
	proj = torch.eye(dim)-jac @ torch.linalg.solve(pjac @ jac, pjac)

	r = reach_est(x_out, proj, radius, encoder, decoder, cloud_size)
	return x_out, r




def reach_est(x, proj_jac, l2_error, encoder, decoder,cloud_size):
	n_points,dim = x.shape
	u = torch.randn(n_points,cloud_size,dim)
	u = u/torch.norm(u, dim = 2).unsqueeze(2)*(torch.rand(n_points,cloud_size,1))**(1.0/dim)*2*l2_error.unsqueeze(1).unsqueeze(1)
	proj = decoder(encoder(u + x.unsqueeze(1))) - x.unsqueeze(1)
	return torch.min((torch.linalg.norm(proj, dim = 2)**2+1e-16)/(2*torch.linalg.norm(torch.bmm(proj_jac,proj.movedim(2,1)), dim = 1)+1e-16), dim = 1).values
