from losses import PrototypicalLoss, MagnetLoss, MagnetLossEval, RepmetLoss#, CrossEntropyLoss


def initialize_loss(config, loss_name, split='train', n_classes=None):
    # losses must return loss, sample_losses, pred, acc
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
            return MagnetLossEval(L=config.test.L)  # give a predictor
        else:
            raise ValueError("Split '%s' not recognised for the %s loss." % (split, loss_name))

    elif loss_name == 'repmet_c':
        assert n_classes is not None
        if split == 'train':
            return RepmetLoss(N=n_classes, k=config.train.k, emb_size=config.model.emb_size,
                              alpha=config.train.alpha, sigma=config.train.sigma, dist=config.model.dist)
        elif split == 'val':
            return RepmetLoss(N=n_classes, k=config.train.k, emb_size=config.model.emb_size,
                              alpha=config.val.alpha, sigma=config.val.sigma, dist=config.model.dist)
        elif split == 'test':
            return RepmetLoss(N=n_classes, k=config.train.k, emb_size=config.model.emb_size,
                              alpha=config.test.alpha, sigma=config.test.sigma, dist=config.model.dist)
        else:
            raise ValueError("Split '%s' not recognised for the %s loss." % (split, loss_name))
    elif loss_name == 'ce':
        # assumes output is a softmax
        # TODO implemento
        raise NotImplementedError
        # return CrossEntropyLoss(softmaxed=False)

