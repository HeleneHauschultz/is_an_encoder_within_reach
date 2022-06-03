import torch
from AutoEncoder import AutoEncoder
from stochman import nnj
import argparse
import time
from torch.nn.functional import softplus
from torch.nn.utils import clip_grad_norm_




def encoder(data_dim, latent_dim):
	encoder = nnj.Sequential(
		nnj.Linear(data_dim, 500), nnj.ELU(),
		nnj.Linear(500,250), nnj.ELU(),
		nnj.Linear(250,150), nnj.ELU(),
		nnj.Linear(150,100), nnj.ELU(),
		nnj.Linear(100,50), nnj.ELU(),
		nnj.Linear(50,2)
		)
	return encoder

def decoder(data_dim, latent_dim):
	decoder = nnj.Sequential(
		nnj.Linear(2,50), nnj.ELU(),
		nnj.Linear(50,100), nnj.ELU(),
		nnj.Linear(100,150), nnj.ELU(),
		nnj.Linear(150,250), nnj.ELU(),
		nnj.Linear(250,500), nnj.ELU(),
		nnj.Linear(500,data_dim)
		)
	return decoder

def main(args):
#Loading the data
	X_data = torch.load("picked_MNIST.t")[:5000]
	experiment_name = args.experiment_name
	output_dir = f'out/{experiment_name}'

#Initial training of autoencoder
	ae = AutoEncoder(encoder(784,2), decoder(784,2), lr = 1e-4, output_dir = output_dir)
	ae.do_train(X_data, n_train_iter = args.pretrain_epochs)
	#ae.lr = 1e-5
	#ae.do_train(X_data, n_train_iter = 20_000)
	ae.save(f"{experiment_name}_ae_trained.pt")

#Saving the loss curve
	l2_r = ae.l2_reach(X_data, cloud_size = 10)
	torch.save(l2_r, f"out/{experiment_name}/{experiment_name}_initial_reach")
	l2_error = torch.norm(X_data- ae(X_data))
	print(torch.sum(l2_error > l2_r))

	reach_points = []
	reach_losses = []
	l2_losses = []
 
#Reach training

	for j in range(args.reach_epochs):
		iter_start_time = time.time()

    #Reconstructed points with jacobians
		x_out , jac = ae.decoder(ae.encoder(X_data), jacobian = True)
		l2_error = torch.linalg.norm(x_out-X_data, dim = 1)

		l2_loss = ((x_out-X_data)**2).mean()

    #Calculating projection matrices
		pjac = torch.permute(jac, (0,2,1))
		proj_jac = torch.eye(784)- jac @ torch.linalg.solve(pjac@jac, pjac)

		r = torch.zeros(len(X_data))

    #Calculating reach in batches
		n_points = 500
		for i in range(10):
			u = torch.randn(n_points,1000,784)
			u = u/torch.norm(u, dim = 2).unsqueeze(2)*(torch.rand(n_points,1000,1))**(1.0/784.0)*2*l2_error[i*n_points:(i+1)*n_points].unsqueeze(1).unsqueeze(1)
			proj = ae(u + x_out[i*n_points:(i+1)*n_points].unsqueeze(1)) - x_out[i*n_points:(i+1)*n_points].unsqueeze(1)
			r[i*n_points:(i+1)*n_points] = torch.min((torch.linalg.norm(proj, dim = 2)**2+1e-16)/(2*torch.linalg.norm(torch.bmm(proj_jac[i*n_points:(i+1)*n_points],proj.movedim(2,1)), dim = 1)+1e-16), dim = 1).values

    #Calculating the reach loss
		reach_loss = softplus(l2_error - r).mean()
		print(f"Iteration {j} #points outside reach: {torch.sum(l2_error>r)}")
		reach_points.append(torch.sum(l2_error>r))

		loss = l2_loss + reach_loss
		reach_losses.append(reach_loss.item())
		l2_losses.append(l2_loss.item())
   
		print(f"l2_loss: {l2_loss.item()}, reach_loss: {reach_loss.item()}")

    #Optimizing
		assert torch.isfinite(loss), "non-finite loss"
		ae.optim.zero_grad()
		loss.backward()
		#for p in ae.parameters():
		#	assert torch.all(torch.isfinite(p.grad)), "non-finite gradients"

		clip_grad_norm_(ae.parameters(), 10.0)
		ae.optim.step()
   





  #Save model and loss curves
	ae.save(f"{experiment_name}_reach_trained.pt")
	torch.save(reach_points, f"out/{experiment_name}/reach_points.pt")
	torch.save(reach_losses, f"out/{experiment_name}/reach_losses.pt")
	torch.save(l2_losses, f"out/{experiment_name}/l2_losses_iter.pt")

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = "MNIST auto-encoder with reach regularization")
	parser.add_argument('--experiment_name', default = 'reach_reg_ae_mnist', type=str, help='name for directories and files')
	parser.add_argument('--reach_epochs', default = 40, type = int, help = 'number of epochs of reach regularization')
	parser.add_argument('--pretrain_epochs', default = 5000, type = int, help = 'number of epochs of pretraining')
	args = parser.parse_args()
	main(args)

