from torchvision import transforms, datasets

from data_loading.sets.oxford_flowers import OxFlowers
from data_loading.stanford_dogs import StanDogs


def load_datasets(set_id, set_path, input_size=224, stats=True):
    """
    Function for loading the datasets we want to use, this will also allow merged sets in future,
    all data returned will be transformed accordingly
    :param set_id:
    :param set_path:
    :param input_size:
    :param stats:
    :return:
    """
    train_dataset = None
    val_dataset = None
    test_dataset = None

    if set_id == 'mnist':
        train_dataset = datasets.MNIST(root=set_path,
                                       train=True,
                                       transform=transforms.ToTensor(),
                                       download=True)
        val_dataset = datasets.MNIST(root=set_path,
                                      train=False,
                                      transform=transforms.ToTensor())

    elif set_id == 'stanford_dogs':
        input_transforms = transforms.Compose([
            transforms.RandomResizedCrop(input_size, ratio=(1, 1.3)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])

        train_dataset = StanDogs(root=set_path,
                                 train=True,
                                 cropped=False,
                                 transform=input_transforms,
                                 download=True)
        val_dataset = StanDogs(root=set_path,
                                train=False,
                                cropped=False,
                                transform=input_transforms,
                                download=True)

        if stats:
            print("Training set stats:")
            train_dataset.stats()
            print("Validation set stats:")
            val_dataset.stats()

    elif set_id == 'oxford_flowers':
        input_transforms = transforms.Compose([
            transforms.RandomResizedCrop(input_size, ratio=(1, 1.3)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])

        train_dataset = OxFlowers(root=set_path,
                                  train=True,
                                  val=False,
                                  transform=input_transforms,
                                  download=True)
        val_dataset = OxFlowers(root=set_path,
                                train=False,
                                val=True,
                                transform=input_transforms,
                                download=True)

        if stats:
            print("Training set stats:")
            train_dataset.stats()
            print("Validation set stats:")
            val_dataset.stats()

    return train_dataset, val_dataset, test_dataset