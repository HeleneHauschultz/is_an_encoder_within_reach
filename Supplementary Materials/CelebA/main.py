from tasks.celeba.model import CelebA

if __name__ == '__main__':
    model = CelebA()
    model.load('../969454d/latest')
    model.compute_reach()
