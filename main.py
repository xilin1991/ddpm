import os
import sys
import datetime
import argparse

# %matplotlib inline
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from loguru import logger
import matplotlib.pyplot as plt

from model.diffusion import MLPDiffusion
from dataset.point_dataset import PointDataset


def diffusion_loss_fn(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
    batch_size = x_0.shape[0]

    t = torch.randint(0, n_steps, size=(batch_size // 2,))
    t = torch.cat([t, n_steps - 1 - t], dim=0)
    t = t.unsqueeze(-1).cuda()

    a = alphas_bar_sqrt[t]

    am1 = one_minus_alphas_bar_sqrt[t]

    e = torch.randn_like(x_0)

    x = x_0 * a + e * am1

    output = model(x, t.squeeze(-1))

    return (e - output).square().mean()


def p_sample_loop(model, shape, n_steps, betas, one_minus_alphas_bar_sqrt):

    cur_x = torch.randn(shape, device=betas.device)
    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model, cur_x, i, betas, one_minus_alphas_bar_sqrt)
        x_seq.append(cur_x)

    return x_seq


def p_sample(model, x, t, betas, one_minus_alphas_bar_sqrt):

    t = torch.tensor([t], device=betas.device)

    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]

    eps_theta = model(x, t)

    mean = (1 / (1 - betas[t]).sqrt()) * (x - (coeff * eps_theta))

    z = torch.randn_like(x)
    sigma_t = betas[t].sqrt()

    sample = mean + sigma_t * z

    return (sample)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', default='b', type=str, help='dataset type: [s, o, b, m, r]')
    parser.add_argument('--num_epoch', default=64000, type=int, help='total training epoch')
    parser.add_argument('--batch_size', default=1024, type=int, help='training batch size')
    parser.add_argument('--num_steps', default=100, type=int, help='add noise steps')

    args = parser.parse_args()

    logger_format = '<g>{time:MMM-DD HH:mm}</g> | {level} | {message}'
    logger.remove()
    logger.add(sys.stdout, format=logger_format)

    logger.info(f'Curve: {str(args.data_type).upper()}')

    exp_id = '%s_%s' % (datetime.datetime.now().strftime('%b%d_%H.%M.%S'), str(args.data_type).upper())
    res_dirs = f'results/{exp_id}'
    os.makedirs(res_dirs, exist_ok=True)

    dataset = PointDataset(data_type=args.data_type, res_dirs=res_dirs)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = MLPDiffusion(args.num_steps).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    betas = torch.linspace(-6, 6, args.num_steps).cuda()
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, 0)
    alphas_prod_p = torch.cat([torch.tensor([1], device=betas.device).float(), alphas_prod[:-1]], 0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    assert alphas.shape == alphas_prod.shape == alphas_prod_p.shape == \
        alphas_bar_sqrt.shape == one_minus_alphas_bar_log.shape == \
            one_minus_alphas_bar_sqrt.shape
    logger.info(f'all the same shape: {betas.shape}')
    # print("all the same shape:", betas.shape)
    plt.rc('text', color='blue')

    for t in range(args.num_epoch):
        for idx, batch_x in enumerate(dataloader):
            loss = diffusion_loss_fn(model, batch_x.cuda(), alphas_bar_sqrt, one_minus_alphas_bar_sqrt, args.num_steps)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()

        if (t % 100 == 0):
            # print(loss.item())
            logger.info(f'epoch-{t}: {loss.item()}')
            x_seq = p_sample_loop(model, dataset.points.shape, args.num_steps, betas, one_minus_alphas_bar_sqrt)

            fig, axs = plt.subplots(1, 10, figsize=(28, 3))
            for i in range(1, 11):
                cur_x = x_seq[i * 10].detach().cpu()
                axs[i - 1].scatter(cur_x[:, 0], cur_x[:, 1], color='red', edgecolor='white')
                axs[i - 1].set_axis_off()
                axs[i - 1].set_title(r'$q(\mathbf{x}_{'+str(i*10)+'})$')

            plt.savefig(f'{res_dirs}/epoch-{t}.png', dpi=600, bbox_inches='tight')
            plt.close()
