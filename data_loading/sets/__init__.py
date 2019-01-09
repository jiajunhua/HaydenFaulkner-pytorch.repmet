# This allows us to import the datasets without having to refer to their python file
from data_loading.sets.omniglot import OmniglotDataset
from data_loading.sets.oxford_flowers import OxfordFlowersDataset
from data_loading.sets.oxford_pets import OxfordPetsDataset
from data_loading.sets.stanford_dogs import StanfordDogsDataset
from data_loading.sets.pascal_voc import PascalVOCDataset
from data_loading.sets.combined import CombinedDataset