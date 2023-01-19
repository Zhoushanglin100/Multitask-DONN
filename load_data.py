import os, warnings
import torch
import torchvision
from torchvision.datasets.vision import VisionDataset
from PIL import Image

class EMNIST(VisionDataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    training_file = 'training_letters.pt'
    test_file = 'test_letters.pt'
    classes = ['10 - ten', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine',
               '11 - elven', '12 - twelve','13 - thirteen', '14 - fourteen', 
               '15 - fifteen', '16 - sixteen', '17 - seventeen', '18 - eighteen',
               '19 - ninteen', '20 - twenty', '21 - twentyone', '22 - twentytwo',
               '23 - twentythree', '24 - twentyfour', '25 - twentyfive', '26 - twentysix' ]

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, divide=0, train=True, transform=None, target_transform=None,
                 download=False):
        super(EMNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train  # training set or test set
        self.root = root
        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

        idx0 = self.targets == 10
        idx1 = self.targets == 1
        idx2 = self.targets == 2
        idx3 = self.targets == 3
        idx4 = self.targets == 4
        idx5 = self.targets == 5
        idx6 = self.targets == 6
        idx7 = self.targets == 7
        idx8 = self.targets == 8
        idx9 = self.targets == 9
        idx10 = self.targets == 11
        idx11 = self.targets == 12
        idx12 = self.targets == 13
        idx13 = self.targets == 14
        idx14 = self.targets == 15
        idx15 = self.targets == 16
        idx16 = self.targets == 17
        idx17 = self.targets == 18
        idx18 = self.targets == 19
        idx19 = self.targets == 20
        idx20 = self.targets == 21
        idx21 = self.targets == 22
        idx22 = self.targets == 23
        idx23 = self.targets == 24
        idx24 = self.targets == 25
        idx25 = self.targets == 26
        idx_0_9 = torch.cat((idx0.nonzero(),idx1.nonzero(),idx2.nonzero(),idx3.nonzero(),idx4.nonzero(),idx5.nonzero(), idx6.nonzero(), idx7.nonzero(), idx8.nonzero(), idx9.nonzero()),dim=0).view(-1)
        idx_10_19 = torch.cat((idx10.nonzero(),idx11.nonzero(),idx12.nonzero(),idx13.nonzero(),idx14.nonzero(), idx15.nonzero(), idx16.nonzero(), idx17.nonzero(), idx18.nonzero(), idx19.nonzero()),dim=0).view(-1)
        if divide==0:
            self.data = self.data[idx_0_9]
            self.targets = self.targets[idx_0_9] - 1
        elif divide==1:
            self.data = self.data[idx_10_19]
            self.targets = self.targets[idx_10_19]
        else:
            assert(0), "wrong split option"
        # print(self.data.shape, self.targets.shape)
        # print(np.unique(self.targets))
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.test_file)))

    def download(self):
        """Download the eMNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        # download files
        train_set = torchvision.datasets.EMNIST("./Edata", split='letters', download=True,train=True, transform=self.transform)
        test_set = torchvision.datasets.EMNIST("./Edata", split='letters', download=True, train=False, transform=self.transform)
        print('Done!')

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")