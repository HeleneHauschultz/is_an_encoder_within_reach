import torch as t
import tqdm


class Model(t.nn.Module):

    def train_batch(self) -> float:
        raise NotImplemented()

    def eval_batch(self) -> float:
        raise NotImplemented()

    def save(self, fn):
        t.save({
            'batch_idx': self.batch_idx,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, fn)

    def load(self, fn):
        checkpoint = t.load(fn, map_location=t.device(self.device))
        self.batch_idx = checkpoint["batch_idx"]
        self.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def do_train(self, n_updates=int(1e6), eval_interval=1000):
        best = float("inf")
        for i in tqdm.tqdm(range(n_updates)):
            self.train_batch()
            if (i + 1) % eval_interval == 0:
                loss = self.eval_batch()
                self.save("latest")
                if loss < best:
                    best = loss
                    self.save("best")
