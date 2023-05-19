from registry import register
from functools import partial
from pytorch_metric_learning import losses
# from pytorch_metric_learning.distances import LpDistance, DotProductSimilarity
import torch
import torch.nn.functional as F
registry = {}
register = partial(register, registry=registry)

@register('ntx')
class NTXLoss():
    def __init__(self, **kwargs):
        t = 0.07
        if 't' in kwargs:
            t = kwargs['t']
        self.criterion = losses.NTXentLoss(temperature=t)

    def __call__(self, target, production, device):
        embeddings = torch.cat([target, production], dim=0)

        labels = torch.arange(embeddings.size()[0] // 2)
        labels = torch.cat([labels, labels], dim=0)
        labels.to(device)
        return self.criterion(embeddings, labels)


def lalign(x, y, alpha=2):
    return (x - y).norm(dim=1).pow(alpha).mean()


def lunif(x, t=2):
    sq_pdist = torch.pdist(x, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean().log()


@register('align_uniform')
class Align_uniform():
    def __init__(self, alpha=2, t=2, **kwargs):
        self.lam = 1.0
        if 'lam' in kwargs:
            self.lam = kwargs['lam']
        self.alpha = alpha
        self.t = t

    def __call__(self, x, y, device):
        # target_factor = l2normfactor(x)
        # x, y = x/target_factor, y/target_factor
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        align_loss_val = lalign(x, y, self.alpha)
        unif_loss_val = (lunif(x, self.t) + lunif(y, self.t)) / 2
        # embeddings = torch.cat([x, y], dim=0)
        # unif_loss_val = lunif(embeddings, self.t)
        loss = align_loss_val + self.lam * unif_loss_val
        print(align_loss_val.item(), unif_loss_val.item())
        return loss
