import os

import imageio
import numpy as np
import torch
from sklearn.metrics import jaccard_score
from torch import nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as v_utils
from dataset.cityscapes_video import cityscapesLoader
from models.unet_rnn import UNet

writer = SummaryWriter()
def log_metrics(phase, epoch, loss, accuracy):
    # create a tag for the metrics
    tag_loss = f'{phase}/loss'
    tag_accuracy = f'{phase}/accuracy'

    # write the metrics to the summary writer
    writer.add_scalar(tag_loss, loss, epoch)
    writer.add_scalar(tag_accuracy, accuracy, epoch)


def quantitative_metrics(phase, epoch, loss, accuracy):
    # create a tag for the metrics
    tag_loss = f'{phase}/MIoU'
    tag_accuracy = f'{phase}/accuracy'

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

    return y_, y, x


def main():
    # hyper-parameters (learning rate and how many epochs we will train for)
    lr = 0.0002
    epochs = 75
    validation_interval = 5
    augmented = False

    # cityscapes dataset loading
    root = '/home/nfs/inf6/data/datasets/cityscapes/'
    train_data = cityscapesLoader(root=root, split='train', is_transform=False, img_size=(256, 512))
    img_batch = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True, num_workers=4, drop_last=True)

    root = '/home/nfs/inf6/data/datasets/cityscapes/'
    val_data = cityscapesLoader(root=root, split='val', is_transform=False, img_size=(512, 1024))
    img_batch_val = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=True, num_workers=4, drop_last=True)

    # initiate generator
    print("creating unet model...")
    generator = nn.DataParallel(UNet(3, 20), device_ids=[i for i in range(1)]).cuda()

    # start the training and validation
    train(generator, img_batch, train_data, img_batch_val, epochs, lr, validation_interval, augmented, num_classes=20)

    writer.close()


if __name__ == '__main__':
    main()
