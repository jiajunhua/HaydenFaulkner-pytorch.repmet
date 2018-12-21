from losses import PrototypicalLoss, MagnetLoss, MagnetLossEval


def initialize_loss(config, loss_name, split='train'):

    if loss_name == 'prototypical':
        if split == 'train':
            return PrototypicalLoss(n_support=config.train.support_per_epi)
        elif split == 'val':
            return PrototypicalLoss(n_support=config.val.support_per_epi)
        elif split == 'test':
            return PrototypicalLoss(n_support=config.test.support_per_epi)
        else:
            raise ValueError("Split '%s' not recognised for the %s loss." % (split, loss_name))

    elif loss_name == 'magnet':
        if split == 'train':
            return MagnetLoss(m=config.train.m, d=config.train.d, alpha=1.0)
        elif split == 'val':
            return MagnetLossEval(L=config.val.L)  # give it a predictor
        elif split == 'test':
            return MagnetLossEval(L=config.test.L)
            # give a predictor
        else:
            raise ValueError("Split '%s' not recognised for the %s loss." % (split, loss_name))

