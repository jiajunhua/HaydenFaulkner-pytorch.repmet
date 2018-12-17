from losses.prototypical_loss import PrototypicalLoss


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
