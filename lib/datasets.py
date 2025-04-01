import os
import cv2
import json
import torch
import scipy
import scipy.io as sio
from skimage import io
import imageio
import numpy as np
from PIL import Image
from skimage.transform import resize
from skimage import img_as_ubyte, img_as_float32
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

class Flowers(ImageFolder):
    def __init__(self, root, train=True, transform=None, **kwargs):
        self.dataset_root = root
        self.loader = default_loader
        self.target_transform = None
        self.transform = transform
        label_path = os.path.join(root, 'imagelabels.mat')
        split_path = os.path.join(root, 'setid.mat')

        print('Dataset Flowers is trained with resolution 224!')

        # labels
        labels = sio.loadmat(label_path)['labels'][0]
        self.img_to_label = dict()
        for i in range(len(labels)):
            self.img_to_label[i] = labels[i]

        splits = sio.loadmat(split_path)
        self.trnid, self.valid, self.tstid = sorted(splits['trnid'][0].tolist()), \
                                             sorted(splits['valid'][0].tolist()), \
                                             sorted(splits['tstid'][0].tolist())
        if train:
            self.imgs = self.trnid + self.valid
        else:
            self.imgs = self.tstid

        self.samples = []
        for item in self.imgs:
            self.samples.append((os.path.join(root, 'jpg', "image_{:05d}.jpg".format(item)), self.img_to_label[item-1]-1))

class Cars196(ImageFolder, datasets.CIFAR10):
    base_folder_devkit = 'devkit'
    base_folder_trainims = 'cars_train'
    base_folder_testims = 'cars_test'

    filename_testanno = 'cars_test_annos.mat'
    filename_trainanno = 'cars_train_annos.mat'

    base_folder = 'cars_train'
    train_list = [
        ['00001.jpg', '8df595812fee3ca9a215e1ad4b0fb0c4'],
        ['00002.jpg', '4b9e5efcc3612378ec63a22f618b5028']
    ]
    test_list = []
    num_training_classes = 98 # 196/2

    def __init__(self, root, train=False, transform=None, target_transform=None, **kwargs):
        self.root = root
        self.transform = transform

        self.target_transform = target_transform
        self.loader = default_loader
        print('Dataset Cars196 is trained with resolution 224!')

        self.samples = []
        self.nb_classes = 196

        if train:
            labels = \
            sio.loadmat(os.path.join(self.root, self.base_folder_devkit, self.filename_trainanno))['annotations'][0]
            for item in labels:
                img_name = item[-1].tolist()[0]
                label = int(item[4]) - 1
                self.samples.append((os.path.join(self.root, self.base_folder_trainims, img_name), label))
        else:
            labels = \
            sio.loadmat(os.path.join(self.root, 'cars_test_annos_withlabels.mat'))['annotations'][0]
            for item in labels:
                img_name = item[-1].tolist()[0]
                label = int(item[-2]) - 1
                self.samples.append((os.path.join(self.root, self.base_folder_testims, img_name), label))

class Pets(ImageFolder):
    def __init__(self, root, train=True, transform=None, target_transform=None, **kwargs):
        self.dataset_root = root
        self.loader = default_loader
        self.target_transform = None
        self.transform = transform
        train_list_path = os.path.join(self.dataset_root, 'annotations', 'trainval.txt')
        test_list_path = os.path.join(self.dataset_root, 'annotations', 'test.txt')

        self.samples = []
        if train:
            with open(train_list_path, 'r') as f:
                for line in f:
                    img_name = line.split(' ')[0]
                    label = int(line.split(' ')[1])
                    self.samples.append((os.path.join(root, 'images', "{}.jpg".format(img_name)), label-1))
        else:
            with open(test_list_path, 'r') as f:
                for line in f:
                    img_name = line.split(' ')[0]
                    label = int(line.split(' ')[1])
                    self.samples.append((os.path.join(root, 'images', "{}.jpg".format(img_name)), label-1))

class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

class UCMDataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, seed=42, resize_to=(224, 224), fold_idx=3):
        self.seed = seed
        self.size = resize_to
        self.num_channels = 3
        self.num_classes = 21
        self.root_dir = root
        self.transform = transform
        self.fold_idx = fold_idx
        self.N = 2100
        self.train = train
        self._load_data()

    def _load_data(self):
        images = np.zeros((self.N, self.size[0], self.size[1], self.num_channels), dtype="uint8")
        labels = []
        filenames = []

        i = 0
        for item in os.listdir(self.root_dir):
            item_path = os.path.join(self.root_dir, item)
            if os.path.isdir(item_path):
                for subitem in os.listdir(item_path):
                    subitem_path = os.path.join(item_path, subitem)
                    filenames.append(subitem_path)
                    image = imageio.imread(subitem_path)
                    if image.shape[:2] != self.size:
                        image = img_as_ubyte(resize(image, self.size, anti_aliasing=True))
                    images[i] = image #images[i] = img_as_ubyte(image)
                    labels.append(item)
                    i += 1

        labels = np.asarray(labels)
        filenames = np.asarray(filenames)

        sorted_indices = filenames.argsort()
        images = images[sorted_indices]
        labels = labels[sorted_indices]

        le = preprocessing.LabelEncoder()
        le.fit(np.sort(np.unique(labels)))
        labels = le.transform(labels)
        labels = np.asarray(labels)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        splits = list(skf.split(images, labels))
        train_idx, test_idx = splits[self.fold_idx]

        self.train_data, self.test_data = images[train_idx], images[test_idx]
        self.train_labels, self.test_labels = labels[train_idx], labels[test_idx]

        if self.train:
            self.data = self.train_data
            self.targets = self.train_labels
        else:
            self.data = self.test_data
            self.targets = self.test_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = Image.fromarray(self.data[idx])
        if self.transform:
            img = self.transform(img)

        return img, self.targets[idx]

class UCMDataset50(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, seed=42, resize_to=(224, 224), train_ratio=0.5):
        self.seed = seed
        self.size = resize_to
        self.num_channels = 3
        self.num_classes = 21
        self.root_dir = root
        self.transform = transform
        self.N = 2100
        self.train = train
        self.train_ratio = train_ratio
        self._load_data()

    def _load_data(self):
        images = np.zeros((self.N, self.size[0], self.size[1], self.num_channels), dtype="uint8")
        labels = []
        filenames = []

        i = 0
        for item in os.listdir(self.root_dir):
            item_path = os.path.join(self.root_dir, item)
            if os.path.isdir(item_path):
                for subitem in os.listdir(item_path):
                    subitem_path = os.path.join(item_path, subitem)
                    filenames.append(subitem_path)
                    image = imageio.imread(subitem_path)
                    if image.shape[:2] != self.size:
                        image = img_as_ubyte(resize(image, self.size, anti_aliasing=True))
                    images[i] = image
                    labels.append(item)
                    i += 1

        labels = np.asarray(labels)
        filenames = np.asarray(filenames)

        # 按文件名排序以确保数据一致
        sorted_indices = filenames.argsort()
        images = images[sorted_indices]
        labels = labels[sorted_indices]

        # 将类别转换为数值编码
        le = preprocessing.LabelEncoder()
        le.fit(np.sort(np.unique(labels)))
        labels = le.transform(labels)

        # 50%:50% 划分数据
        train_data, test_data, train_labels, test_labels = train_test_split(
            images, labels, train_size=self.train_ratio, stratify=labels, random_state=self.seed
        )

        self.data = train_data if self.train else test_data
        self.targets = train_labels if self.train else test_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = Image.fromarray(self.data[idx])
        if self.transform:
            img = self.transform(img)

        return img, self.targets[idx]

class NWPUDataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, seed=42, resize_to=(224, 224), fold_idx=0):
        self.seed = seed
        self.size = resize_to
        self.num_channels = 3
        self.num_classes = 45
        self.root_dir = root
        self.transform = transform
        self.fold_idx = fold_idx
        self.N = 31500
        self.train = train
        self._load_data()

    def _load_data(self):
        images = np.zeros((self.N, self.size[0], self.size[1], self.num_channels), dtype="uint8")
        labels = []
        filenames = []

        i = 0
        for item in os.listdir(self.root_dir):
            item_path = os.path.join(self.root_dir, item)
            if os.path.isdir(item_path):
                for subitem in os.listdir(item_path):
                    subitem_path = os.path.join(item_path, subitem)
                    filenames.append(subitem_path)
                    image = imageio.imread(subitem_path)
                    if image.shape[:2] != self.size:
                        image = img_as_ubyte(resize(image, self.size, anti_aliasing=True))
                    images[i] = image #images[i] = img_as_ubyte(image)
                    labels.append(item)
                    i += 1

        labels = np.asarray(labels)
        filenames = np.asarray(filenames)

        sorted_indices = filenames.argsort()
        images = images[sorted_indices]
        labels = labels[sorted_indices]

        le = preprocessing.LabelEncoder()
        le.fit(np.sort(np.unique(labels)))
        labels = le.transform(labels)
        labels = np.asarray(labels)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        splits = list(skf.split(images, labels))
        train_idx, test_idx = splits[self.fold_idx]

        self.train_data, self.test_data = images[train_idx], images[test_idx]
        self.train_labels, self.test_labels = labels[train_idx], labels[test_idx]

        if self.train:
            self.data = self.train_data
            self.targets = self.train_labels
        else:
            self.data = self.test_data
            self.targets = self.test_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = Image.fromarray(self.data[idx])
        if self.transform:
            img = self.transform(img)

        return img, self.targets[idx]

class NWPUDataset10(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, seed=42, resize_to=(224, 224), train_ratio=0.1):
        self.seed = seed
        self.size = resize_to
        self.num_channels = 3
        self.num_classes = 45
        self.root_dir = root
        self.transform = transform
        self.N = 31500
        self.train = train
        self.train_ratio = train_ratio
        self._load_data()

    def _load_data(self):
        images = np.zeros((self.N, self.size[0], self.size[1], self.num_channels), dtype="uint8")
        labels = []
        filenames = []

        i = 0
        for item in os.listdir(self.root_dir):
            item_path = os.path.join(self.root_dir, item)
            if os.path.isdir(item_path):
                for subitem in os.listdir(item_path):
                    subitem_path = os.path.join(item_path, subitem)
                    filenames.append(subitem_path)
                    image = imageio.imread(subitem_path)
                    if image.shape[:2] != self.size:
                        image = img_as_ubyte(resize(image, self.size, anti_aliasing=True))
                    images[i] = image
                    labels.append(item)
                    i += 1

        labels = np.asarray(labels)
        filenames = np.asarray(filenames)

        # 按文件名排序以确保数据一致
        sorted_indices = filenames.argsort()
        images = images[sorted_indices]
        labels = labels[sorted_indices]

        # 将类别转换为数值编码
        le = preprocessing.LabelEncoder()
        le.fit(np.sort(np.unique(labels)))
        labels = le.transform(labels)

        # 50%:50% 划分数据
        train_data, test_data, train_labels, test_labels = train_test_split(
            images, labels, train_size=self.train_ratio, stratify=labels, random_state=self.seed
        )

        self.data = train_data if self.train else test_data
        self.targets = train_labels if self.train else test_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = Image.fromarray(self.data[idx])
        if self.transform:
            img = self.transform(img)

        return img, self.targets[idx]


class AIDDataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, seed=42, resize_to=(224, 224), fold_idx=4):
        self.seed = seed
        self.size = resize_to
        self.num_channels = 3
        self.num_classes = 30
        self.root_dir = root
        self.transform = transform
        self.fold_idx = fold_idx
        self.N = 10000
        self.train = train
        self._load_data()

    def _load_data(self):
        images = np.zeros((self.N, self.size[0], self.size[1], self.num_channels), dtype="uint8")
        labels = []
        filenames = []

        i = 0
        for item in os.listdir(self.root_dir):
            item_path = os.path.join(self.root_dir, item)
            if os.path.isdir(item_path):
                for subitem in os.listdir(item_path):
                    subitem_path = os.path.join(item_path, subitem)
                    filenames.append(subitem_path)
                    image = imageio.imread(subitem_path)
                    if image.shape[:2] != self.size:
                        image = img_as_ubyte(resize(image, self.size, anti_aliasing=True))
                    images[i] = image #images[i] = img_as_ubyte(image)
                    labels.append(item)
                    i += 1

        labels = np.asarray(labels)
        filenames = np.asarray(filenames)

        sorted_indices = filenames.argsort()
        images = images[sorted_indices]
        labels = labels[sorted_indices]

        le = preprocessing.LabelEncoder()
        le.fit(np.sort(np.unique(labels)))
        labels = le.transform(labels)
        labels = np.asarray(labels)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        splits = list(skf.split(images, labels))
        test_idx, train_idx = splits[self.fold_idx]

        self.train_data, self.test_data = images[train_idx], images[test_idx]
        self.train_labels, self.test_labels = labels[train_idx], labels[test_idx]

        if self.train:
            self.data = self.train_data
            self.targets = self.train_labels
        else:
            self.data = self.test_data
            self.targets = self.test_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = Image.fromarray(self.data[idx])
        if self.transform:
            img = self.transform(img)

        return img, self.targets[idx]

class AIDDataset50(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, seed=42, resize_to=(224, 224), train_ratio=0.5):
        self.seed = seed
        self.size = resize_to
        self.num_channels = 3
        self.num_classes = 30
        self.root_dir = root
        self.transform = transform
        self.N = 10000
        self.train = train
        self.train_ratio = train_ratio
        self._load_data()

    def _load_data(self):
        images = np.zeros((self.N, self.size[0], self.size[1], self.num_channels), dtype="uint8")
        labels = []
        filenames = []

        i = 0
        for item in os.listdir(self.root_dir):
            item_path = os.path.join(self.root_dir, item)
            if os.path.isdir(item_path):
                for subitem in os.listdir(item_path):
                    subitem_path = os.path.join(item_path, subitem)
                    filenames.append(subitem_path)
                    image = imageio.imread(subitem_path)
                    if image.shape[:2] != self.size:
                        image = img_as_ubyte(resize(image, self.size, anti_aliasing=True))
                    images[i] = image
                    labels.append(item)
                    i += 1

        labels = np.asarray(labels)
        filenames = np.asarray(filenames)

        # 按文件名排序以确保数据一致
        sorted_indices = filenames.argsort()
        images = images[sorted_indices]
        labels = labels[sorted_indices]

        # 将类别转换为数值编码
        le = preprocessing.LabelEncoder()
        le.fit(np.sort(np.unique(labels)))
        labels = le.transform(labels)

        # 50%:50% 划分数据
        train_data, test_data, train_labels, test_labels = train_test_split(
            images, labels, train_size=self.train_ratio, stratify=labels, random_state=self.seed
        )

        self.data = train_data if self.train else test_data
        self.targets = train_labels if self.train else test_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = Image.fromarray(self.data[idx])
        if self.transform:
            img = self.transform(img)


def build_dataset(is_train, args, folder_name=None):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR10':
        dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 10
    elif args.data_set == 'CIFAR100':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'CARS':
        dataset = Cars196(args.data_path, train=is_train, transform=transform)
        nb_classes = 196
    elif args.data_set == 'PETS':
        dataset = Pets(args.data_path, train=is_train, transform=transform)
        nb_classes = 37
    elif args.data_set == 'FLOWERS':
        dataset = Flowers(args.data_path, train=is_train, transform=transform)
        nb_classes = 102
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'EVO_IMNET':
        root = os.path.join(args.data_path, folder_name)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'UCM':
        dataset = UCMDataset(args.data_path, train=is_train, transform=transform)
        nb_classes = 21
    elif args.data_set == 'UCM50':
        dataset = UCMDataset50(args.data_path, train=is_train, transform=transform, train_ratio=0.5)
        nb_classes = 21
    elif args.data_set == 'NWPU':
        dataset = NWPUDataset(args.data_path, train=is_train, transform=transform)
        nb_classes = 45
    elif args.data_set == 'NWPU10':
        dataset = NWPUDataset10(args.data_path, train=is_train, transform=transform, train_ratio=0.1)
        nb_classes = 45
    elif args.data_set == 'AID':
        dataset = AIDDataset(args.data_path, train=is_train, transform=transform)
        nb_classes = 30
    elif args.data_set == 'AID50':
        dataset = AIDDataset50(args.data_path, train=is_train, transform=transform, train_ratio=0.5)
        nb_classes = 30

    return dataset, nb_classes

def build_dataset_fold(is_train, args, folder_name=None):
    transform = build_transform(is_train, args)
    if args.data_set == 'UCM':
        dataset = UCMDataset(args.data_path, train=is_train, transform=transform, fold_idx= args.fold_idx)
        nb_classes = 21
    elif args.data_set == 'UCM50':
        dataset = UCMDataset50(args.data_path, train=is_train, transform=transform, train_ratio=0.5)
        nb_classes = 21
    elif args.data_set == 'NWPU':
        dataset = NWPUDataset(args.data_path, train=is_train, transform=transform, fold_idx= args.fold_idx)
        nb_classes = 45
    elif args.data_set == 'MWPU10':
        dataset = NWPUDataset10(args.data_path, train=is_train, transform=transform, train_ratio=0.1)
        nb_classes = 45
    elif args.data_set == 'AID':
        dataset = AIDDataset(args.data_path, train=is_train, transform=transform, fold_idx= args.fold_idx)
        nb_classes = 30
    elif args.data_set == 'AID50':
        dataset = AIDDataset50(args.data_path, train=is_train, transform=transform, train_ratio=0.5)
        nb_classes = 30

    return dataset, nb_classes

def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
