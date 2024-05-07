import torch.nn as nn


class MLPDiffusion(nn.Module):
    """_summary_

    Parameters
    ----------
    nn : _type_
        _description_
    """
    def __init__(self, n_steps, num_units=128):
        """_summary_

        Parameters
        ----------
        n_steps : _type_
            _description_
        num_units : int, optional
            _description_, by default 128
        """
        super(MLPDiffusion, self).__init__()

        self.linears = nn.ModuleList([
            nn.Linear(2, num_units),
            nn.ReLU(),
            nn.Linear(num_units, num_units),
            nn.ReLU(),
            nn.Linear(num_units, num_units),
            nn.ReLU(),
            nn.Linear(num_units, 2)
            ])
        self.step_embeddings = nn.ModuleList([
            nn.Embedding(n_steps, num_units),
            nn.Embedding(n_steps, num_units),
            nn.Embedding(n_steps, num_units)
        ])

    def forward(self, x_0, t):
        """_summary_

        Parameters
        ----------
        x_0 : _type_
            _description_
        t : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        x = x_0
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            x = self.linears[2 * idx](x)
            x += t_embedding
            x = self.linears[2 * idx + 1](x)

        x = self.linears[-1](x)

        return x
