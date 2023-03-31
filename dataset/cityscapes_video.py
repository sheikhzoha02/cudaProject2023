import glob
import os

import numpy as np
import torch
import torch.utils.data as data
from PIL.Image import Image
from torchvision.transforms import transforms


class cityscapesLoader(data.Dataset):
    colors = [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
        [255, 255, 255]

    ]

    # makes a dictionary with key:value. For example 0:[128, 64, 128]
    label_colours = dict(zip(range(19), colors))

    def __init__(
            self,
            root,
            # which data split to use
            split="train",
            # transform function activation
            is_transform=False,
            # image_size to use in transform function
            img_size=(256, 512),
            augment=False,
            sequence_length=5
    ):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.leftImg8bit_sequence_path = os.path.join(root, f"leftImg8bit_sequence/{split}")
        self.gtFine_sequence_path = os.path.join(root, f"gtFine_sequence/{split}")
        self.augment = augment
        self.image_files = sorted(glob.glob(os.path.join(self.leftImg8bit_sequence_path, '*/*_leftImg8bit.png')))
        self.label_files = sorted(glob.glob(os.path.join(self.gtFine_sequence_path, '*/*_gtFine_labelIds.png')))
        self.tuples = []
        self.final_tuples = []
        split_counter_sequential = 15
        split_counter_augmented = 10

        if (split == 'val' and sequence_length == 12):
            split_counter_sequential = 10
            split_counter_augmented = 1

        for i in range(len(self.label_files)):
            self.final_tuples.append(
                (self.image_files[split_counter_sequential + (i * 30):(i * 30) + 20], self.label_files[i]))

        self.final_tuples_augmented = []
        final_skip_augmented = []
        for i in range(len(self.label_files)):
            new_image_files = self.image_files[split_counter_augmented + (i * 30):(i * 30) + 21]
            for j in range(len(new_image_files)):
                if j % 2 == 1:
                    final_skip_augmented.append(new_image_files[j])
            self.final_tuples_augmented.append((final_skip_augmented, self.label_files[i]))

        self.n_classes = 19
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.files = {}
        self.sequence_length = sequence_length

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]

        # these are 19
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33,
                              ]

        # these are 19 + 1; "unlabelled" is extra
        self.class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]

        # for void_classes; useful for loss function
        self.ignore_index = 19

        # dictionary of valid classes 7:0, 8:1, 11:2
        self.class_map = dict(zip(self.valid_classes, range(19)))

    def __len__(self):
        return len(self.final_tuples)

    def __getitem__(self, index):
        sequential_augmented_frames = []

        transform_img = transforms.Compose([
            transforms.Resize(size=(self.img_size[0], self.img_size[1]), interpolation=Image.BILINEAR),
            transforms.ToTensor()
        ])

        transform_lbl = transforms.Compose([
            transforms.Resize(size=(self.img_size[0], self.img_size[1]), interpolation=Image.NEAREST)
        ])

        # path of image
        img_path, lbl_path = self.final_tuples[index]
        # read image
        img = [(Image.open(img_path[i]).convert('RGB')) for i in range(self.sequence_length)]
        # convert to numpy array
        img = [transform_img(img[i]) for i in range(self.sequence_length)]

        # read label
        lbl = (Image.open(lbl_path).convert('L'))
        lbl = transform_lbl(lbl)
        lbl_np = np.array(lbl)
        # encode using encode_segmap function: 0...18 and 250
        lbl_np = self.encode_segmap(lbl_np)
        lbl_rgb = self.decode_segmap(lbl_np)

        # convert the lbl_rgb numpy array to a PyTorch tensor
        lbl_rgb_tensor = torch.from_numpy(lbl_rgb.transpose(2, 0, 1).astype(np.float32))
        lbl_rgb_tensor /= 255.0

        # Convert label back to a PyTorch tensor
        lbl_tensor = torch.from_numpy(lbl_np).long()

        sequential_augmented_frames.append((img, lbl_tensor, lbl_rgb_tensor))

        # path of image
        img_path, lbl_path = self.final_tuples_augmented[index]
        # read image
        img = [(Image.open(img_path[i]).convert('RGB')) for i in range(self.sequence_length)]
        # convert to numpy array
        img = [transform_img(img[i]) for i in range(self.sequence_length)]

        # read label
        lbl = (Image.open(lbl_path).convert('L'))
        lbl = transform_lbl(lbl)
        lbl_np = np.array(lbl)
        # encode using encode_segmap function: 0...18 and 250
        lbl_np = self.encode_segmap(lbl_np)
        lbl_rgb = self.decode_segmap(lbl_np)

        # convert the lbl_rgb numpy array to a PyTorch tensor
        lbl_rgb_tensor = torch.from_numpy(lbl_rgb.transpose(2, 0, 1).astype(np.float32))
        lbl_rgb_tensor /= 255.0

        # Convert label back to a PyTorch tensor
        lbl_tensor = torch.from_numpy(lbl_np).long()
        sequential_augmented_frames.append((img, lbl_tensor, lbl_rgb_tensor))
        return sequential_augmented_frames

    def decode_segmap(self, temp):

        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        return rgb

    # there are different class 0...33
    # we are converting that info to 0....18; and 250 for void classes
    # final mask has values 0...18 and 250
    def encode_segmap(self, mask):
        # !! Comment in code had wrong informtion
        # Put all void classes to ignore_index
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask