import sys
import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms.functional as TF
from collections import namedtuple
import torch.nn as nn
import glob
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.transforms import ToTensor
import glob
import torchvision.transforms.functional as TF
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, Resize
from sklearn.metrics import jaccard_score
import torch.utils.data as data
import torch.utils as utils
import torch.nn.init as init
import torchvision.utils as v_utils
import torchvision.datasets as dset
from torch.autograd import Variable
import torchvision.models as models
import imageio
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

# initialize the summary writer
writer = SummaryWriter()

class ConvLSTM(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, stride):
        super(ConvLSTM, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=in_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            bias=True)

    def forward(self, x, hidden=None):
        if hidden is None:
            batch_size, _, height, width = x.size()
            hidden = (
                torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device),
                torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device))
        hx, cx = hidden
        combined = torch.cat([x, hx], dim=1)
        gates = self.conv(combined)
        i, f, g, o = torch.split(gates, self.hidden_channels, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        cy = f * cx + i * g
        hy = o * torch.tanh(cy)
        return hy, cy


def decode_segmap_to_rgb(lbl_rgb):
    lbl_rgb_tensor = torch.from_numpy(lbl_rgb.transpose(2, 0, 1).astype(np.float32))
    lbl_rgb_tensor /= 255.0
    return lbl_rgb_tensor

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
        [255,255,255]
        
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
            img_size=(256,512),
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
        split_final_counter_augmented = 21
        
        if(split == 'val' and sequence_length == 12):
            split_counter_sequential = 8
            split_counter_augmented = 1
            split_final_counter_augmented = 23

        for i in range(len(self.label_files)):
            self.final_tuples.append((self.image_files[split_counter_sequential + (i * 30):(i * 30) + 20], self.label_files[i]))

        print(len(self.final_tuples))
        self.final_tuples_augmented = []
        final_skip_augmented = []
        for i in range(len(self.label_files)):
            new_image_files = self.image_files[split_counter_augmented + (i * 30):(i * 30) + split_final_counter_augmented]
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


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels , in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class ResUNet34(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(ResUNet34, self).__init__()

        self.in_dim = in_channels
        self.out_dim = out_channels

        self.prev_rnn1 = None
        self.prev_rnn2 = None
        self.prev_rnn3 = None
        self.prev_rnn4 = None
        # self.prev_rnn5 = None

        # Load ResNet34 and extract the encoder blocks
        self.encoder = models.resnet34(pretrained=True)
        self.enc0 = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu) # 1/2
        self.enc1 = nn.Sequential(self.encoder.maxpool, self.encoder.layer1) # 1/4
        self.enc2 = self.encoder.layer2 # 1/8
        self.enc3 = self.encoder.layer3 # 1/16
        self.enc4 = self.encoder.layer4 # 1/32

        self.lstm = ConvLSTM(512, 512, 3, 1)
        self.lstm1 = ConvLSTM(256, 256, 3, 1)
        self.lstm2 = ConvLSTM(128, 128, 3, 1)
        self.lstm3 = ConvLSTM(64, 64, 3, 1)
        self.lstm4 = ConvLSTM(20, 20, 3, 1)

        # Upsample blocks
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 64)

        # Final convolution layer
        self.final_conv = nn.ConvTranspose2d(64, self.out_dim, kernel_size=2, stride=2)

    def forward(self, input, val=False):
        # Encoder path
        images = []
        for input_ in input:
            if val:
                input_ = F.interpolate(input_, size=(512, 1024), mode='bilinear', align_corners=True)            

            x0 = self.enc0(input_)
            x1 = self.enc1(x0)
            x2 = self.enc2(x1)
            x3 = self.enc3(x2)
            x4 = self.enc4(x3)

            lstm = self.lstm(x4, self.prev_rnn1)
            detached_hidden_state = (lstm[0].detach(), lstm[1].detach())
            lstm1 = self.lstm1(x3, self.prev_rnn2)
            detached_hidden_state1 = (lstm1[0].detach(), lstm1[1].detach())
            lstm2 = self.lstm2(x2, self.prev_rnn3)
            detached_hidden_state2 = (lstm2[0].detach(), lstm2[1].detach())
            lstm3 = self.lstm3(x1, self.prev_rnn4)
            detached_hidden_state3 = (lstm3[0].detach(), lstm3[1].detach())


            # Decoder path
            x = self.up1(lstm[0], lstm1[0])
            x = self.up2(x, lstm2[0])
            x = self.up3(x, lstm3[0])
            x0 = x0.narrow(dim=1, start=0, length=32)
            x = self.up4(x,x0)

            # Final layer
            x = self.final_conv(x)
            if val:
                x = F.interpolate(x, size=(1024,2048), mode='bilinear', align_corners=True)

            # lstm4 = self.lstm4(x, self.prev_rnn5)
            # detached_hidden_state4 = (lstm4[0].detach(), lstm4[1].detach())
            # self.final_rnn_conv = nn.Conv2d(20, self.out_dim, kernel_size=3, padding=1).cuda()
            # x = self.final_rnn_conv(lstm4[0])

            images.append(x)
            self.prev_rnn1, self.prev_rnn2, self.prev_rnn3, self.prev_rnn4 = detached_hidden_state, detached_hidden_state1, detached_hidden_state2, detached_hidden_state3
        return images

def log_metrics(phase, epoch, loss, accuracy):
    # create a tag for the metrics
    tag_loss = f'{phase}/loss Resunet34 with 12 sequence frame'
    tag_accuracy = f'{phase}/accuracy Resunet34 with 12 sequence frame'
    
    # write the metrics to the summary writer
    writer.add_scalar(tag_loss, loss, epoch)
    writer.add_scalar(tag_accuracy, accuracy, epoch)

def quantitative_metrics(phase, epoch, loss, accuracy):
    # create a tag for the metrics
    tag_loss = f'{phase}/MIoU Resunet34 with 12 sequence frame'
    tag_accuracy = f'{phase}/accuracy Resunet34 with 12 sequence frame'
    
    # write the metrics to the summary writer
    writer.add_scalar(tag_loss, loss, epoch)
    writer.add_scalar(tag_accuracy, accuracy, epoch)


def save_checkpoint(model, optimizer, epoch, mean_iou_sequential, mean_accuracy_sequential,
            mean_loss_sequential,train_mean_iou_sequential, train_mean_accuracy_sequential,train_mean_loss_sequential,
            checkpoint_path):
    
    file_loss = open('./unet_loss_34', 'a')
    
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_mean_iou_sequential': mean_iou_sequential,
        'val_mean_accuracy_sequential': mean_accuracy_sequential,
        # 'val_mean_iou_augmented': mean_iou_augmented,
        # 'val_mean_accuracy_augmented': mean_accuracy_augmented,
        'val_mean_loss_sequential': mean_loss_sequential,
        # 'val_mean_loss_augmented': mean_loss_augmented,
        'train_mean_iou_sequential': train_mean_iou_sequential,
        'train_mean_accuracy_sequential': train_mean_accuracy_sequential,
        # 'train_mean_iou_augmented': train_mean_iou_augmented,
        # 'train_mean_accuracy_augmented': train_mean_accuracy_augmented,
        'train_mean_loss_sequential': train_mean_loss_sequential,
        # 'train_mean_loss_augmented': train_mean_loss_augmented
    }

    file_loss.write("val_mean_iou_sequential : " + str(mean_iou_sequential)+"\n")
    file_loss.write("val_mean_accuracy_sequential : " + str(mean_accuracy_sequential)+"\n")
    file_loss.write("val_mean_loss_sequential : " + str(mean_loss_sequential)+"\n")
    file_loss.write("train_mean_iou_sequential : " + str(train_mean_iou_sequential)+"\n")
    file_loss.write("train_mean_accuracy_sequential : " + str(train_mean_accuracy_sequential)+"\n")
    file_loss.write("train_mean_loss_sequential : " + str(train_mean_loss_sequential)+"\n")

    log_metrics('train', epoch, train_mean_loss_sequential, train_mean_accuracy_sequential)
    log_metrics('val', epoch, mean_loss_sequential, mean_accuracy_sequential)
    quantitative_metrics('train', epoch, train_mean_iou_sequential, train_mean_accuracy_sequential)
    quantitative_metrics('val', epoch, mean_iou_sequential, mean_accuracy_sequential)

    torch.save(state, checkpoint_path)

def calc_jaccard_score(y_true, y_pred):

    batch_size, heights, width = y_true.shape
    num_classes = y_pred.shape[1]

    # Convert y_pred to the same format as y_true by selecting the class with the highest probability
    y_pred_argmax = y_pred.argmax(dim=1)


    y_true_np = y_true.cpu().numpy().astype(int)
    y_pred_argmax_np = y_pred_argmax.cpu().numpy().astype(int)

    jaccard_scores = []
    for i in range(batch_size):
        jaccard_scores.append(jaccard_score(y_true_np[i].ravel(), y_pred_argmax_np[i].ravel(), average='weighted'))

    mean_jaccard = np.mean(jaccard_scores)

    print(f"Mean Jaccard score: {mean_jaccard}")
    return mean_jaccard


def mean_pixel_accuracy(outputs, targets, num_classes):

    outputs = torch.argmax(outputs, dim=1)
    # Flatten the outputs and targets
    outputs = outputs.view(-1)
    targets = targets.view(-1)

    # Count the number of correct predictions
    correct = torch.sum(outputs == targets)

    # Compute the total number of pixels
    total_pixels = outputs.shape[0]

    # Compute the mean pixel accuracy
    mean_accuracy = correct / total_pixels

    return mean_accuracy

def check_valid_class_indices(tensor, n_classes):
    min_class_index = tensor.min().item()
    max_class_index = tensor.max().item()
    print(f'min:{min_class_index} max:{max_class_index}')
    if min_class_index < 0 or max_class_index >= n_classes:
        return False
    return True

def compute_iou_and_accuracy(y_true, y_pred, num_classes):
    iou_list = []
    accuracy_list = []
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)

    for cls in range(num_classes):
        intersection = (y_true == cls) & (y_pred == cls)
        union = (y_true == cls) | (y_pred == cls)
        correct = (y_true == y_pred) & (y_true == cls)

        iou = torch.true_divide(intersection.sum().float(), union.sum().float())
        accuracy = torch.true_divide(correct.sum().float(), (y_true == cls).sum().float())

        iou_list.append(iou.item())
        accuracy_list.append(accuracy.item())

    return iou_list, accuracy_list


def save_tensors_as_gif(tensor_list, save_path, duration=0.1):
    images = [tensor.permute(1, 2, 0).numpy() for tensor in tensor_list]
    imageio.mimsave(save_path, images, duration=duration)


def train(generator, img_batch, img_data, img_batch_val, epochs, lr, validation_interval,augmented, num_classes=20):
    train_iou_scores_sequential = []
    train_accuracy_scores_sequential = []
    train_gif_images_sequential = []
    train_loss_sequential_array = []

    train_iou_scores_augmented = []
    train_accuracy_scores_augmented = []
    train_gif_images_augmented = []
    train_loss_augmented_array = []

    train_mean_iou_augmented = 0
    train_mean_accuracy_augmented = 0
    train_mean_loss_augmented = 0

    train_mean_iou_sequential = 0
    train_mean_accuracy_sequential = 0
    train_mean_loss_sequential = 0

    recon_loss_func = nn.CrossEntropyLoss()

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    
    file_loss = open('./unet_loss', 'a')

    # if not os.path.exists('./final_training_resnet34/'):
    #     os.makedirs('./final_training_resnet34/')

    # if not os.path.exists('./final_training_resnet34_label/'):
    #     os.makedirs('./final_training_resnet34_label/')

    # if not os.path.exists('./final_training_resnet34_original/'):
    #     os.makedirs('./final_training_resnet34_original/')

    if not os.path.exists('./models_resnet34/'):
        os.makedirs('./models_resnet34/')

    # if not os.path.exists('./final_gif_images_resnet34/'):
    #     os.makedirs('./final_gif_images_resnet34/')    

    for epoch in range(epochs):
        generator.train()
        for idx_batch, (sequential_frames, skip_frames) in enumerate(img_batch):
            gen_optimizer.zero_grad()
            x_sequential = [Variable(i.float()).cuda(0) for i in sequential_frames[0]]  # Permute the dimensions
            y_sequential = Variable(sequential_frames[1].long()).cuda(0)
            y = generator.forward(x_sequential)

            for result in y:
                iou, accuracy = compute_iou_and_accuracy(y_sequential, result.max(1)[1], num_classes)
                train_iou_scores_sequential.append(iou)
                train_accuracy_scores_sequential.append(accuracy)


            loss = recon_loss_func(y[-1], y_sequential)
            train_loss_sequential_array.append(loss.item())
            # file_loss.write(str(loss.item())+"\n")
            loss.backward()
            gen_optimizer.step()

            if idx_batch % 400 == 0:
                print("Sequential frame epoch = "+str(epoch)+" | loss = "+str(loss.item()))
                sequential_frames[2] = sequential_frames[2].float()

            if(augmented == True):
                gen_optimizer.zero_grad()          
                x_augmented = [Variable(i.float()).cuda(0) for i in skip_frames[0]]  # Permute the dimensions
                y_augmented = Variable(skip_frames[1].long()).cuda(0)
                y = generator.forward(x_augmented)

                for result in y:
                    iou, accuracy = compute_iou_and_accuracy(y_augmented, result.max(1)[1], num_classes)
                    train_iou_scores_augmented.append(iou)
                    train_accuracy_scores_augmented.append(accuracy)


                loss = recon_loss_func(y[-1], y_augmented)
                train_loss_augmented_array.append(loss.item())
                # file_loss.write(str(loss.item())+"\n")
                loss.backward()
                gen_optimizer.step()

                if idx_batch % 400 == 0:
                    print("Augmented frame epoch = "+str(epoch)+" | loss = "+str(loss.item()))
                    skip_frames[2] = skip_frames[2].float()

            train_mean_iou_sequential = np.nanmean(np.nanmean(train_iou_scores_sequential, axis=0))
            train_mean_accuracy_sequential = np.nanmean(np.nanmean(train_accuracy_scores_sequential, axis=0))
            train_mean_loss_sequential = np.nanmean(np.nanmean(train_loss_sequential_array, axis=0))

        # if(augmented == True):
        #     train_mean_iou_augmented = np.nanmean(np.nanmean(train_iou_scores_augmented, axis=0))
        #     train_mean_accuracy_augmented = np.nanmean(np.nanmean(train_accuracy_scores_augmented, axis=0))
        #     train_mean_loss_augmented = np.nanmean(np.nanmean(train_loss_augmented_array, axis=0))

        # if epoch % validation_interval == 0:
        mean_iou_sequential,mean_accuracy_sequential,mean_loss_sequential = validate(generator, img_batch_val, img_data, epoch,augmented, num_classes=20)
        save_checkpoint(generator, gen_optimizer, epoch, mean_iou_sequential, mean_accuracy_sequential,
            mean_loss_sequential, train_mean_iou_sequential, train_mean_accuracy_sequential,
            train_mean_loss_sequential, f'models_resnet34/{epoch}_{idx_batch}.pth')


def validate(generator, img_batch_val,img_data,epoch,augmented, num_classes=20):
    generator.eval()
    recon_loss_func = nn.CrossEntropyLoss()

    iou_scores_sequential = []
    accuracy_scores_sequential = []
    gif_images_sequential = []
    loss_sequential_array = []

    iou_scores_augmented = []
    accuracy_scores_augmented = []
    gif_images_augmented = []
    loss_augmented_array = []

    mean_iou_augmented = 0
    mean_accuracy_augmented = 0
    mean_loss_augmented = 0

    mean_iou_sequential = 0
    mean_accuracy_sequential = 0
    mean_loss_sequential = 0

    with torch.no_grad():
        for idx_batch, (sequential_frames, skip_frames) in enumerate(img_batch_val):
                y_,y,x = iterate_validation_images(generator, sequential_frames[0],sequential_frames[1],sequential_frames[2])
                for result in y:
                    iou, accuracy = compute_iou_and_accuracy(y_, result.max(1)[1], num_classes)
                    iou_scores_sequential.append(iou)
                    accuracy_scores_sequential.append(accuracy)

                loss_sequential = recon_loss_func(y[-1], y_)
                loss_sequential_array.append(loss_sequential.item())

                # generated_image = [img_data.decode_segmap(np.squeeze(i.data.max(1)[1][0, :, :].cpu().numpy())) for i in y]
                # generated_image = [decode_segmap_to_rgb(i) for i in generated_image]
                # gif_images_sequential = [(torch.tensor(i)) for i in generated_image]
                # [v_utils.save_image(i,f"final_training_resnet34/generated_sequential_{epoch}_{ind}.png") for ind,i in enumerate(generated_image)]
                # v_utils.save_image(sequential_frames[2], f"final_training_resnet34_label/label_image_{epoch}.png")
                # save_tensors_as_gif(gif_images_sequential,f"final_gif_images_resnet34/generated_sequential_{epoch}.gif",0.1)

                # # gif_images_sequential = [(torch.tensor(i).squeeze().cpu()) for i in x]
                # [v_utils.save_image(i,f"final_training_resnet34_original/original_sequential_{epoch}_{ind}.png") for ind,i in enumerate(x)]
                # save_tensors_as_gif(gif_images_sequential,f"final_training_resnet34/original_sequential_{epoch}.gif",0.1)

            # if(augmented == True):
            #     y_,y,x = iterate_validation_images(generator, skip_frames[0],skip_frames[1],skip_frames[2])
            #     loss_augmented = recon_loss_func(y[-1], y_)
            #     loss_augmented_array.append(loss_augmented.item())
            #     for result in y:
            #         iou, accuracy = compute_iou_and_accuracy(y_, result.max(1)[1], num_classes)
            #         iou_scores_augmented.append(iou)
            #         accuracy_scores_augmented.append(accuracy)
                
            #     generated_image = [img_data.decode_segmap(np.squeeze(i.data.max(1)[1][0, :, :].cpu().numpy())) for i in y]
            #     generated_image = [decode_segmap_to_rgb(i) for i in generated_image]
            #     gif_images_augmented = [(torch.tensor(i)) for i in generated_image]
            #     [v_utils.save_image(i,f"final_training_resnet34/generated_augmented_{epoch}_{ind}.png") for ind,i in enumerate(generated_image)]
            #     v_utils.save_image(skip_frames[2], f"final_training_resnet34_label/label_image_{epoch}.png")
            #     save_tensors_as_gif(gif_images_augmented,f"final_gif_images_resnet34/generated_augmented_{epoch}.gif",0.1)

            #     # gif_images_augmented = [(torch.tensor(i).squeeze().cpu()) for i in x]
            #     [v_utils.save_image(i,f"final_training_resnet34_original/original_augmented_{epoch}_{ind}.png") for ind,i in enumerate(x)]
            #     # save_tensors_as_gif(gif_images_augmented,f"final_training_resnet34/original_augmented_{epoch}.gif",0.1)

        mean_iou_sequential = np.nanmean(np.nanmean(iou_scores_sequential, axis=0))
        mean_accuracy_sequential = np.nanmean(np.nanmean(accuracy_scores_sequential, axis=0))
        mean_loss_sequential = np.nanmean(np.nanmean(loss_sequential_array, axis=0))

    # if(augmented == True):
    #     mean_iou_augmented = np.nanmean(np.nanmean(iou_scores_augmented, axis=0))
    #     mean_accuracy_augmented = np.nanmean(np.nanmean(accuracy_scores_augmented, axis=0))
    #     mean_loss_augmented = np.nanmean(np.nanmean(loss_augmented_array, axis=0))

    print(f"Validation Sequential frames mean IoU: {mean_iou_sequential}")
    print(f"Validation Sequential frames mean Accuracy: {mean_accuracy_sequential}")

    # print(f"Validation Augmented frames mean IoU: {mean_iou_augmented}")
    # print(f"Validation Augmented frames mean Accuracy: {mean_accuracy_augmented}")

    print(f"Validation Loss Sequential: {mean_loss_sequential}")
    # print(f"Validation Loss Augmented: {mean_loss_augmented}")

    return mean_iou_sequential,mean_accuracy_sequential,mean_loss_sequential


def iterate_validation_images(generator, imagergb, labelmask, labelrgb):    
    x = [Variable(i.float()).cuda(0) for i in imagergb]  # Permute the dimensions
    y_ = Variable(labelmask.long()).cuda(0)
    y = generator.forward(x, val=True)

    return y_,y,x


def main():
    # hyper-parameters (learning rate and how many epochs we will train for)
    lr = 0.0002
    epochs = 100
    validation_interval = 5
    augmented = True

    # cityscapes dataset loading
    root = '/home/nfs/inf6/data/datasets/cityscapes/'
    train_data = cityscapesLoader(root=root, split='train',is_transform=False, img_size=(512,1024))
    img_batch = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True, num_workers=4, drop_last=True)

    root = '/home/nfs/inf6/data/datasets/cityscapes/'
    val_data = cityscapesLoader(root=root, split='val',is_transform=False, img_size=(1024,2048),augment=False,sequence_length=12)
    img_batch_val = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=True, num_workers=4, drop_last=True)

    # initiate generator
    print("creating unet model...")
    generator = nn.DataParallel(ResUNet34(3, 20), device_ids=[i for i in range(1)]).cuda()


    # start the training and validation
    train(generator, img_batch, train_data, img_batch_val, epochs, lr, validation_interval, augmented, num_classes=20)

    writer.close()

if __name__ == '__main__':
    main()

