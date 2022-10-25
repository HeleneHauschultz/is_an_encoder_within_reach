import torch
from AutoEncoder import AutoEncoder
from stochman import nnj
import argparse
import time
from torch.nn.functional import softplus
from torch.nn.utils import clip_grad_norm_
import os

def encoder(data_dim, latent_dim):
    encoder = nnj.Sequential(
        nnj.Linear(data_dim, 500),
        nnj.ELU(),
        nnj.Linear(500, 250),
        nnj.ELU(),
        nnj.Linear(250, 150),
        nnj.ELU(),
        nnj.Linear(150, 100),
        nnj.ELU(),
        nnj.Linear(100, 50),
        nnj.ELU(),
        nnj.Linear(50, latent_dim),
    )
    return encoder


def decoder(data_dim, latent_dim):
    decoder = nnj.Sequential(
        nnj.Linear(latent_dim, 50),
        nnj.ELU(),
        nnj.Linear(50, 100),
        nnj.ELU(),
        nnj.Linear(100, 150),
        nnj.ELU(),
        nnj.Linear(150, 250),
        nnj.ELU(),
        nnj.Linear(250, 500),
        nnj.ELU(),
        nnj.Linear(500, data_dim),
    )
    return decoder

#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device("cpu")

def main(args):
    # Loading the data
    amount = 250 if args.data_size == 'small' else 5000
    X_data = torch.load("picked_MNIST.t")[:amount]
    
    experiment_name = f"{args.experiment_name}_alpha_{args.alpha}"
    base_name = f'out_final_{args.latent_dim}_{args.squared_reach}'
    output_dir = f"{base_name}/{experiment_name}"

    # Initial training of autoencoder
    ae = AutoEncoder(encoder(784, args.latent_dim), decoder(784, args.latent_dim), lr=1e-4, output_dir=output_dir)
    if f"reach_reg_ae_mnist_alpha_ae_trained.pt" in os.listdir(base_name):
        ae.load(f"{base_name}/reach_reg_ae_mnist_alpha_ae_trained.pt")
        print('load model')
    else:
        ae.do_train(X_data, n_train_iter=args.pretrain_epochs)
        ae.save(f"reach_reg_ae_mnist_alpha_ae_trained.pt")
        os.rename(
            f"{output_dir}/reach_reg_ae_mnist_alpha_ae_trained.pt",
            f"{base_name}/reach_reg_ae_mnist_alpha_ae_trained.pt"
        )

    # Saving the loss curve
    l2_r = ae.l2_reach(X_data, cloud_size=10)
    torch.save(l2_r, f"{base_name}/{experiment_name}/{experiment_name}_initial_reach")
    l2_error = torch.norm(X_data - ae(X_data))
    print(torch.sum(l2_error > l2_r))

    reach_points = []
    reach_losses = []
    l2_losses = []

    ae = ae.to(device)
    X_data = X_data.to(device)

    # Reach optimizer
    decoder_optim = torch.optim.Adam(ae.decoder.parameters(), lr=1e-4)
    encoder_optim = torch.optim.Adam(ae.encoder.parameters(), lr=1e-4)

    # Reach training
    for j in range(args.reach_epochs):
        # Decoder optimization
        # Reconstructed points with jacobians
        x_out, jac = ae.decoder(ae.encoder(X_data), jacobian=True)
        l2_error = torch.linalg.norm(x_out - X_data, dim=1)

        l2_loss = ((x_out - X_data) ** 2).mean()

        # Calculating projection matrices
        pjac = torch.permute(jac, (0, 2, 1))
        proj_jac = torch.eye(784, device=device) - jac @ torch.linalg.solve(pjac @ jac, pjac)

        r = torch.zeros(len(X_data), device=device)

        # Calculating reach in batches
        n_batches = 10
        n_points = X_data.shape[0] // n_batches
        print(n_batches, n_points)
        for i in range(n_batches):
            u = torch.randn(n_points, 1000, 784, device=device)
            u = (
                u
                / torch.norm(u, dim=2).unsqueeze(2)
                * (torch.rand(n_points, 1000, 1, device=device)) ** (1.0 / 784.0)
                * 2
                * l2_error[i * n_points : (i + 1) * n_points].unsqueeze(1).unsqueeze(1)
            )
            proj = ae(
                u + x_out[i * n_points : (i + 1) * n_points].unsqueeze(1)
            ) - x_out[i * n_points : (i + 1) * n_points].unsqueeze(1)
            r[i * n_points : (i + 1) * n_points] = torch.min(
                (torch.linalg.norm(proj, dim=2) ** 2 + 1e-16)
                / (
                    2
                    * torch.linalg.norm(
                        torch.bmm(
                            proj_jac[i * n_points : (i + 1) * n_points],
                            proj.movedim(2, 1),
                        ),
                        dim=1,
                    )
                    + 1e-16
                ),
                dim=1,
            ).values

        # Calculating the reach loss
        diff = l2_error - r
        reach_loss = softplus(diff * diff if args.squared_reach else diff).mean()
        print(f"Iteration {j} #points outside reach: {torch.sum(l2_error>r)}")
        reach_points.append(torch.sum(l2_error > r))

        # loss = l2_loss + reach_loss
        reach_losses.append(reach_loss.item())
        l2_losses.append(l2_loss.item())

        print(f"l2_loss: {l2_loss.item()}, reach_loss: {reach_loss.item()}")

        # Reach optimizing
        assert torch.isfinite(reach_loss), "non-finite reach loss"
        decoder_optim.zero_grad()
        (args.alpha*reach_loss).backward()

        # Encoder optimization
        x_out = ae(X_data)
        l2_loss = ((x_out - X_data) ** 2).mean()

        assert torch.isfinite(l2_loss), "non-finite loss"
        encoder_optim.zero_grad()
        l2_loss.backward()
        clip_grad_norm_(ae.parameters(), 1.0)
        encoder_optim.step()
        decoder_optim.step()

        if (j % 10) == 0:
            torch.save(reach_points, f"{base_name}/{experiment_name}/reach_points_{j}.pt")
            torch.save(reach_losses, f"{base_name}/{experiment_name}/reach_losses_{j}.pt")
            torch.save(l2_losses, f"{base_name}/{experiment_name}/l2_losses_iter_{j}.pt")
            ae.save(f"{experiment_name}_reach_trained_{j}")

    # Save model and loss curves
    ae.save(f"{experiment_name}_reach_trained.pt")
    torch.save(reach_points, f"{base_name}/{experiment_name}/reach_points.pt")
    torch.save(reach_losses, f"{base_name}/{experiment_name}/reach_losses.pt")
    torch.save(l2_losses, f"{base_name}/{experiment_name}/l2_losses_iter.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MNIST auto-encoder with reach regularization"
    )
    parser.add_argument(
        "--experiment_name",
        default="reach_reg_ae_mnist",
        type=str,
        help="name for directories and files",
    )
    parser.add_argument(
        "--reach_epochs",
        default=40,
        type=int,
        help="number of epochs of reach regularization",
    )
    parser.add_argument(
        "--pretrain_epochs",
        default=5000,
        type=int,
        help="number of epochs of pretraining",
    )
    parser.add_argument("--alpha", default=1, type=float, help="reach reg constant")
    parser.add_argument("--latent_dim", default=2, type=int, help="latent dimension")
    parser.add_argument('--squared_reach', action='store_true')
    parser.add_argument("--data_size", default='small', type=str, help='choose between small|large')
    args = parser.parse_args()
    main(args)
