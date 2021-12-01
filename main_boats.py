import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm.notebook import tqdm
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt

from configs.VOCconfig import cfg
from models.batch_GAIN_VOC_mutilabel_singlebatch import batch_GAIN_VOC_multiheatmaps
from utils import deprocess_image
from utils.image import show_cam_on_image, denorm


def main():
    data_dir = '/content/drive/MyDrive/boat_data/'
    classes = os.listdir(data_dir)
    print(classes)
    print(len(classes))

    train_transform = transforms.Compose([
        #transforms.RandomRotation(10),  # rotate +/- 10 degrees
        #transforms.RandomHorizontalFlip(),  # reverse 50% of images
        transforms.Resize(100),  # resize shortest side to 100 pixels
        transforms.CenterCrop(100),  # crop longest side to 100 pixels at center
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    orig_transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Resize(100),  # resize shortest side to 100 pixels
        transforms.CenterCrop(100),  # crop longest side to 100 pixels at center
    ])


    orig_dataset = ImageFolder(data_dir, transform=orig_transform)
    dataset = ImageFolder(data_dir, transform=train_transform)

    torch.manual_seed(10)
    val_size = len(dataset) // 10
    test_size = len(dataset) // 5
    train_size = len(dataset) - val_size - test_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
    torch.manual_seed(torch.initial_seed())
    orig_train_ds, orig_val_ds, test_ds = random_split(orig_dataset, [train_size, val_size, test_size])
    len(train_ds), len(val_ds), len(test_ds)

    img, label = dataset[100]
    print(img.shape)

    def show_image(img, label):
        print('Label: ', dataset.classes[label], "(" + str(label) + ")")
        plt.imshow(img.permute(1, 2, 0))

    show_image(*dataset[20])

    batch_size = 1
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)
    torch.manual_seed(torch.initial_seed())
    orig_train_loader = DataLoader(orig_train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)
    torch.manual_seed(torch.initial_seed())
    val_loader = DataLoader(val_ds, batch_size, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size, num_workers=0, pin_memory=True)

    for images, labels in train_loader:
        fig, ax = plt.subplots(figsize=(18, 10))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        break

    def get_default_device():
        """Pick GPU if available, else CPU"""
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        else:
            return torch.device('cpu')

    def to_device(data, device):
        """Move tensor(s) to chosen device"""
        if isinstance(data, (list, tuple)):
            return [to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)

    class DeviceDataLoader():
        """Wrap a dataloader to move data to a device"""

        def __init__(self, dl, device):
            self.dl = dl
            self.device = device

        def __iter__(self):
            """Yield a batch of data after moving it to device"""
            torch.manual_seed(torch.initial_seed())
            for b in self.dl:
                yield to_device(b, self.device)

        def __len__(self):
            """Number of batches"""
            return len(self.dl)

    device = get_default_device()
    print(device)

    train_loader = DeviceDataLoader(train_loader, device)
    val_loader = DeviceDataLoader(val_loader, device)
    test_loader = DeviceDataLoader(test_loader, device)

    orig_train_loader = DeviceDataLoader(orig_train_loader, device)
    #orig_val_loader = DeviceDataLoader(val_loader, device)
    #orig_test_loader = DeviceDataLoader(test_loader, device)

    num_classes = 3

    device_name = 'cuda:0'
    device = torch.device(device_name)

    from torchvision.models import vgg19
    model = vgg19(pretrained=True).train().to(device)

    # change the last layer for finetuning
    classifier = model.classifier
    num_ftrs = classifier[-1].in_features
    new_classifier = torch.nn.Sequential(*(list(model.classifier.children())[:-1]),
                                         nn.Linear(num_ftrs, num_classes).to(device))
    model.classifier = new_classifier
    model.train()

    epochs = 50
    loss_fn = torch.nn.BCEWithLogitsLoss()
    lr = 0.00001
    npretrain = 100
    test_first = False
    grads_off = False
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # a GAIN model which saves the chosen classification model and calculates
    # the gradients w.r.t the grad_layer and performs GAIN algorithm
    gain = batch_GAIN_VOC_multiheatmaps(model=model, grad_layer='features',
                                        num_classes=num_classes,
                                        pretraining_epochs=npretrain,
                                        test_first=test_first,
                                        grads_off=bool(grads_off),
                                        device=device)

    cl_factor = 1
    am_factor = 1

    record_itr_train = 50
    record_itr_test = 50

    mean = cfg.MEAN
    std = cfg.STD

    i = 0
    num_train_samples = 0
    logging_path = './logs/'
    logging_name = "boats"
    from torch.utils.tensorboard import SummaryWriter
    import datetime
    writer = SummaryWriter(
        logging_path + logging_name + '_' +
        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    writer.add_text('Start', 'start')

    print('Started')

    for epoch in range(0, epochs):

        total_train_single_accuracy = 0
        total_test_single_accuracy = 0

        epoch_train_cl_loss = 0

        model.train(True)

        if not test_first or (test_first and epoch != 0):

            total_train_single_accuracy = 0

            total_test_single_accuracy = 0

            epoch_train_am_loss = 0
            epoch_train_cl_loss = 0
            epoch_train_total_loss = 0

            for orig_batch, batch in zip(orig_train_loader, train_loader):

                input = batch[0]
                labels = torch.tensor([list(range(3))] * batch_size).to(device)
                one_hot = torch.nn.functional.one_hot(labels).sum(dim=1)

                optimizer.zero_grad()

                logits_cl, logits_am, heatmap, masked_image, mask = gain(input, one_hot)

                class_onehot = torch.stack(tuple(one_hot)).float()

                cl_loss = loss_fn(logits_cl, class_onehot)

                batch_am_labels_scores = []
                for k in range(len(labels[0])):
                    am_scores = nn.Softmax(dim=1)(logits_am[k])
                    am_labels_scores = am_scores.view(-1)[labels[0][k]]
                    batch_am_labels_scores.append(am_labels_scores)

                num_of_labels = 3
                am_loss = sum(batch_am_labels_scores) / num_of_labels

                intersection_loss = 0
                shape = mask[0].size()[2:]
                ccount = 0
                for i1, m1 in enumerate(mask):
                    for i2, m2 in enumerate(mask):
                        if i1 < i2:
                            # only affect high temperature pixels
                            bg1 = torch.zeros_like(m1)
                            high_temperature_indices1 = m1 >= m1.mean()
                            bg1[high_temperature_indices1] = m1[high_temperature_indices1]

                            bg2 = torch.zeros_like(m2)
                            high_temperature_indices2 = m2 >= m2.mean()
                            bg2[high_temperature_indices2] = m2[high_temperature_indices2]
                            intersection_loss += \
                                torch.exp(-((bg1-bg2).abs().sum() /
                                            (high_temperature_indices1.sum()+
                                             high_temperature_indices2.sum())))
                            ccount = ccount + 1
                intersection_loss = intersection_loss / ccount
                total_loss = num_of_labels * cl_loss * cl_factor + am_loss * am_factor# + intersection_loss * 1

                epoch_train_am_loss += (am_loss * am_factor).detach().cpu().item()
                epoch_train_cl_loss += (cl_loss * cl_factor).detach().cpu().item()
                epoch_train_total_loss += total_loss.detach().cpu().item()

                writer.add_scalar('Per_Step/train/cl_loss', (cl_loss * cl_factor).detach().cpu().item(), i)
                writer.add_scalar('Per_Step/train/intersection_loss', (intersection_loss * cl_factor).detach().cpu().item(), i)
                writer.add_scalar('Per_Step/train/am_loss', (am_loss * am_factor).detach().cpu().item(), i)
                writer.add_scalar('Per_Step/train/total_loss', total_loss.detach().cpu().item(), i)

                loss = cl_loss * cl_factor + intersection_loss
                if gain.AM_enabled():
                    loss += am_loss * am_factor
                loss.backward()
                optimizer.step()

                # Accuracy evaluation
                num_of_labels = 3
                _, y_pred = logits_cl[0].detach().topk(k=num_of_labels)
                y_pred = y_pred.view(-1).tolist()
                gt = []
                acc = len(set(y_pred).intersection(set(gt))) / num_of_labels
                total_train_single_accuracy += acc

                if i % record_itr_train == 0:
                    num_of_labels = 3
                    for t in range(num_of_labels):
                        orig = orig_batch[0]
                        orig = orig.permute([0, 2, 3, 1])

                        one_heatmap = heatmap[t].squeeze().cpu().detach().numpy()

                        #one_augmented_im = torch.tensor(np.array(batch[0])).permute([0,2,3,1]).to(device)
                        one_masked_image = masked_image[t].detach().squeeze()
                        htm = deprocess_image(one_heatmap)
                        visualization, red_htm = show_cam_on_image(orig.cpu().detach().numpy(), htm, True)

                        viz = torch.from_numpy(visualization).to(device)
                        masked_im = denorm(one_masked_image, mean, std)
                        masked_im = (masked_im.squeeze().permute([1, 2, 0])
                                     .cpu().detach().numpy() * 255).round() \
                            .astype(np.uint8)


                        masked_im = torch.from_numpy(masked_im).unsqueeze(0).to(device)
                        orig_viz = torch.cat((orig, viz, masked_im), 0)
                        grid = torchvision.utils.make_grid(orig_viz.permute([0, 3, 1, 2]))
                        gt = [str(i) for i in range(3)]
                        writer.add_image(tag='Train_Heatmaps/image_' + str(i) +
                                             '_' + str(t) + '_' + '_'.join(gt),
                                         img_tensor=grid, global_step=epoch,
                                         dataformats='CHW')
                        y_scores = nn.Softmax()(logits_cl[0].detach())
                        _, predicted_categories = y_scores.topk(num_of_labels)
                        predicted_cl = [(gt[x], format(y_scores.view(-1)[x], '.4f')) for x in
                                        predicted_categories.view(-1)]
                        labels_cl = [(gt[x], format(y_scores.view(-1)[x], '.4f')) for x in labels[0]]
                        import itertools
                        predicted_cl = list(itertools.chain(*predicted_cl))
                        labels_cl = list(itertools.chain(*labels_cl))
                        cl_text = 'cl_gt_' + '_'.join(labels_cl) + '_pred_' + '_'.join(predicted_cl)
                        am_scores = nn.Softmax(dim=1)(logits_am[t])
                        _, am_labels = am_scores.topk(num_of_labels)
                        predicted_am = [(gt[x], format(am_scores.view(-1)[x], '.4f')) for x in
                                        am_labels.tolist()[0]]
                        labels_am = [(gt[x], format(am_scores.view(-1)[x], '.4f')) for x in labels[0]]
                        import itertools
                        predicted_am = list(itertools.chain(*predicted_am))
                        labels_am = list(itertools.chain(*labels_am))
                        am_text = '_am_gt_' + '_'.join(labels_am) + '_pred_' + '_'.join(predicted_am)

                        writer.add_text('Train_Heatmaps_Description/image_' + str(i) + '_' + str(t) + '_' +
                                        '_'.join(gt), cl_text + am_text, global_step=epoch)

                i += 1

                if epoch == 0 and test_first == False:
                    num_train_samples += 1
                if epoch == 1 and test_first == True:
                    num_train_samples += 1

        model.train(False)
        j = 0

        for batch in val_loader:

            labels = [0, 1, 2]

            logits_cl, logits_am, heatmap, masked_image, mask = gain(batch, labels)

            num_of_labels = 3

            # Single label evaluation
            _, y_pred = logits_cl.detach().topk(k=num_of_labels)
            y_pred = y_pred.view(-1).tolist()
            gt = labels
            acc = len(set(y_pred).intersection(set(gt))) / num_of_labels
            total_test_single_accuracy += acc

            if j % record_itr_test == 0:
                one_heatmap = heatmap[0].squeeze().cpu().detach().numpy()
                one_input_image = dataset[0].cpu().detach().numpy()
                one_masked_image = masked_image[0].detach().squeeze()
                htm = deprocess_image(one_heatmap)
                visualization, heatmap = show_cam_on_image(one_input_image, htm, True)

                viz = torch.from_numpy(visualization).unsqueeze(0).to(device)
                augmented = torch.tensor(one_input_image).unsqueeze(0).to(device)
                masked_im = denorm(one_masked_image, mean, std)
                masked_im = (masked_im.squeeze().permute([1, 2, 0])
                             .cpu().detach().numpy() * 255).round() \
                    .astype(np.uint8)

                orig = orig_dataset[0].unsqueeze(0)
                masked_im = torch.from_numpy(masked_im).unsqueeze(0).to(device)
                orig_viz = torch.cat((orig, augmented, viz, masked_im), 0)
                grid = torchvision.utils.make_grid(orig_viz.permute([0, 3, 1, 2]))
                gt = [["0", "1", "2"][x] for x in labels[0]]
                writer.add_image(tag='Test_Heatmaps/image_' + str(j) +
                                     '_' + str(0) + '_' + '_'.join(gt),
                                 img_tensor=grid, global_step=epoch,
                                 dataformats='CHW')
                y_scores = nn.Softmax()(logits_cl[0].detach())
                _, predicted_categories = y_scores.topk(num_of_labels)
                predicted_cl = [(gt[x], format(y_scores.view(-1)[x], '.4f')) for x in
                                predicted_categories.view(-1)]
                labels_cl = [(gt[x], format(y_scores.view(-1)[x], '.4f')) for x in labels[0]]
                import itertools
                predicted_cl = list(itertools.chain(*predicted_cl))
                labels_cl = list(itertools.chain(*labels_cl))
                cl_text = 'cl_gt_' + '_'.join(labels_cl) + '_pred_' + '_'.join(predicted_cl)
                am_scores = nn.Softmax(dim=1)(logits_am[0])
                _, am_labels = am_scores.topk(num_of_labels)
                predicted_am = [(gt[x], format(am_scores.view(-1)[x], '.4f')) for x in
                                am_labels.tolist()[0]]
                labels_am = [(gt[x], format(am_scores.view(-1)[x], '.4f')) for x in labels[0]]
                import itertools
                predicted_am = list(itertools.chain(*predicted_am))
                labels_am = list(itertools.chain(*labels_am))
                am_text = '_am_gt_' + '_'.join(labels_am) + '_pred_' + '_'.join(predicted_am)

                writer.add_text('Test_Heatmaps_Description/image_' + str(j) + '_' + str(0) + '_' +
                                '_'.join(gt), cl_text + am_text, global_step=epoch)

            j += 1

        num_test_samples = len(val_loader) * batch_size
        '''
        if len(args.checkpoint_file_path_save) > 0 and epoch % args.checkpoint_nepoch == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iteration': i,
                'num_train_samples': num_train_samples
            }, args.checkpoint_file_path_save + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        '''
        print("finished epoch number:")
        print(epoch)

        if (test_first and epoch > 0) or test_first == False:
            writer.add_scalar('Loss/train/cl_total_loss', epoch_train_cl_loss / (num_train_samples * batch_size), epoch)
            writer.add_scalar('Loss/train/am_tota_loss', epoch_train_am_loss / num_train_samples, epoch)
            writer.add_scalar('Loss/train/combined_total_loss', epoch_train_total_loss / num_train_samples, epoch)
            writer.add_scalar('Accuracy/train/cl_accuracy',
                              total_train_single_accuracy / (num_train_samples * batch_size), epoch)

        writer.add_scalar('Accuracy/test/cl_accuracy', total_test_single_accuracy / num_test_samples, epoch)

        gain.increase_epoch_count()


if __name__ == '__main__':
    main()