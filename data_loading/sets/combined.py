"""
Used to combine multiple sets together, currently only supports detection sets
"""
# todo mod classification sets for support

from torch.utils.data.dataset import Dataset


class CombinedDataset(Dataset):

    def __init__(self,
                 datasets,
                 transform=None,
                 target_transform=None):
        """
        :param datasets: (Dataset) the datasets to combine
        :param transform: how to transform the input, performed after per dataset transforms
        :param target_transform: how to transform the target, performed after per dataset transforms
        """

        super(CombinedDataset, self).__init__()

        # set instance variables
        self.datasets = datasets
        self.transform = transform
        self.target_transform = target_transform

        # setup the categories
        self.categories, self.categories_to_labels, self.labels_to_categories = self._process_categories()
        self.n_categories = len(self.categories)

        # merge the datas
        self.sample_ids = []
        self.sample_ids_sets = {}
        self.data = {}
        for dataset in datasets:
            for sample_id in dataset.data.keys():
                assert sample_id not in self.data  # ensure the sample_ids are unique across sets
                self.sample_ids.append(sample_id)
                self.data[sample_id] = dataset.data[sample_id]
                self.sample_ids_sets[sample_id] = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # get the data sample id
        sample_id = self.sample_ids[index]
        dataset = self.sample_ids_sets[sample_id]

        # load the image
        x = dataset.load_img(sample_id)  # use the dataset object to load img with appropriate function call
        y = self.data[sample_id]

        # perform the transforms
        if self.transform:
            x = self.transform(x)

        if self.target_transform:
            y = self.target_transform(y)

        return x, y

    def get_img_path(self, sample_id):
        return self.sample_ids_sets[sample_id].get_img_path(sample_id)

    def stats(self):
        # get the stats to print
        boxes_p_cls, boxes_p_img = self.class_counts()

        return "# images: %d\n" \
               "# boxes: %d\n"\
               "# categories: %d\n"\
               "Boxes per image (min, avg, max): %d, %d, %d\n"\
               "Boxes per category (min, avg, max): %d, %d, %d\n" % \
               (len(self.data), sum(boxes_p_img), len(boxes_p_cls),
                min(boxes_p_img), sum(boxes_p_img) / len(boxes_p_img), max(boxes_p_img),
                min(boxes_p_cls), sum(boxes_p_cls) / len(boxes_p_cls), max(boxes_p_cls))

    def class_counts(self):
        # calculate the number of samples per category, and per image
        boxes_p_cls = [0]*(self.n_categories-1)  # minus 1 for background removal
        boxes_p_img = []
        for sample_id in self.data.keys():
            boxes_this_img = 0
            annotations = self.data[sample_id]
            for label in annotations['gt_classes']:
                boxes_p_cls[label-1] += 1
                boxes_this_img += 1
            boxes_p_img.append(boxes_this_img)

        return boxes_p_cls, boxes_p_img

    def _process_categories(self):
        # Build categories to labels (cats can be names, labels are ints starting from 0)
        categories = []
        categories_to_labels = {}
        labels_to_categories = {}
        for dataset in self.datasets:
            for c in dataset.categories:  # merges if category name is same
                if c not in categories:
                    categories.append(c)
                    categories_to_labels[c] = len(categories_to_labels)
                    labels_to_categories[categories_to_labels[c]] = c

        return categories, categories_to_labels, labels_to_categories

    @staticmethod
    def display(img, boxes, classes):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        # Display the image
        plt.imshow(img)

        # Add the boxes with labels
        for i in range(len(classes)):
            box = boxes[i]
            plt.gca().add_patch(Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor='r', facecolor='none'))
            plt.text(box[0], box[1]+20, str(classes[i]), fontsize=12, color='r')

        return plt


if __name__ == "__main__":
    # use this for debugging and checks
    from utils.debug import set_working_dir
    from config.config import config
    import matplotlib.pyplot as plt
    from data_loading.sets.pascal_voc import PascalVOCDataset

    # set the working directory as appropriate
    set_working_dir()

    # load the dataset
    datasetA = PascalVOCDataset(root_dir=config.dataset.root_dir, split='train', use_flipped=True)
    datasetB = PascalVOCDataset(root_dir=config.dataset.root_dir, split='val', use_flipped=False)
    datasetC = CombinedDataset(datasets=[datasetA, datasetB])

    # print the stats
    print(datasetC.stats())

    # lets plot some samples
    fig = plt.figure()

    for i in range(len(datasetC)):
        sample = datasetC[i]

        plt = datasetC.display(sample[0], sample[1]['boxes'], sample[1]['gt_classes'])

        plt.show()
        if i == 3:
            break
