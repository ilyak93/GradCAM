import PIL.Image
import pathlib

import argparse
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torchvision.models import vgg19, wide_resnet101_2, mobilenet_v2
import numpy as np
import matplotlib.pyplot as plt
# from torchviz import make_dot
from sys import maxsize as maxint

from dataloaders import data
from dataloaders.MedTData import MedT_Loader
from metrics.metrics import calc_sensitivity
from models.batch_GAIN_VOC_mutilabel_singlebatch import batch_GAIN_VOC_multiheatmaps
from utils.image import show_cam_on_image, preprocess_image, deprocess_image, denorm

from models.batch_GAIN_VOC import batch_GAIN_VOC
from PIL import Image

from torch.utils.tensorboard import SummaryWriter
import datetime

from configs.VOCconfig import cfg

# Parse all the input argument
parser = argparse.ArgumentParser(description='PyTorch GAIN Training')
parser.add_argument('--device', type=str, help='device:ID', default='cpu') #for gpu use the string format of 'cuda:#', e.g: cuda:0
parser.add_argument('--epoch_size', type=int, help='number of iterations per epoch', default='6000')
parser.add_argument('--dataset_path', type=str, help='the path to your local VOC 2012 dataset directory')
parser.add_argument('--logging_path', type=str, help='the path to your local logging directory') #recommended a big enough storage for many runs
parser.add_argument('--logging_name', type=str, help='a name for the current run for logging purposes') #recommended a big enough storage for many runs
parser.add_argument('--checkpoint_file_path_load', type=str, default='', help='a full path including the name of the checkpoint_file to load from, empty otherwise')
parser.add_argument('--checkpoint_file_path_save', type=str, default='', help='a full path including the name of the checkpoint_file to save to, empty otherwise')
parser.add_argument('--checkpoint_nepoch', type=int, help='each how much to save a checkpoint', default=0)
parser.add_argument('--workers_num', type=int, help='number of threads for data loading', default=0)
parser.add_argument('--test_first', type=int, help='0 for not testing before training in the beginning, 1 otherwise', default=0)
parser.add_argument('--cl_loss_factor', type=float, help='a parameter for the classification loss magnitude, constant 1 in the paper', default=1)
parser.add_argument('--am_loss_factor', type=float, help='a parameter for the AM (Attention-Mining) loss magnitude, alpha in the paper', default=1)
parser.add_argument('--nepoch', type=int, help='number of epochs to train', default=50)
parser.add_argument('--lr', type=float, help='learning rate', default=0.00001) #recommended 0.0001 for batchsiuze > 1, 0.00001 otherwise
parser.add_argument('--npretrain', type=int, help='number of epochs to pretrain before using AM', default=5) #recommended at least 1
parser.add_argument('--record_itr_train', type=int, help='each which number of iterations to log images in training mode', default=1000)
parser.add_argument('--record_itr_test', type=int, help='each which number of iterations to log images in test mode', default=100)
parser.add_argument('--nrecord', type=int, help='how much images of a batch to record', default=1)

parser.add_argument('--grads_off', type=int, default=0, help='mode: \
                with gradients (as in the paper) or without \
                (a novel approach of ours)')

def main(args):
    categories = cfg.CATEGORIES
    num_classes = len(categories)

    device = torch.device(args.device)
    model = vgg19(pretrained=True).train().to(device)

    # change the last layer for finetuning
    classifier = model.classifier
    num_ftrs = classifier[-1].in_features
    new_classifier = torch.nn.Sequential(*(list(model.classifier.children())[:-1]),
                                         nn.Linear(num_ftrs, num_classes).to(device))
    model.classifier = new_classifier
    model.train()


    mean = cfg.MEAN
    std = cfg.STD

    batch_size = 1
    batch_size_dict = {'train': batch_size, 'test': batch_size}
    rds = data.RawDataset(root_dir=args.dataset_path,
                          num_workers=args.workers_num,
                          output_dims=cfg.INPUT_DIMS,
                          batch_size_dict=batch_size_dict)

    test_first = bool(args.test_first)

    cl_factor = args.cl_loss_factor
    am_factor = args.am_loss_factor

    epochs = args.nepoch
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # a GAIN model which saves the chosen classification model and calculates
    # the gradients w.r.t the grad_layer and performs GAIN algorithm
    gain = batch_GAIN_VOC_multiheatmaps(model=model, grad_layer='features',
                          num_classes=num_classes,
                          pretraining_epochs=args.npretrain,
                          test_first=test_first,
                          grads_off=bool(args.grads_off),
                          device=device)

    i = 0
    num_train_samples = 0

    chkpnt_epoch = 0
    if len(args.checkpoint_file_path_load) > 0:
        checkpoint = torch.load('args.checkpoint_file_path_load')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        chkpnt_epoch = checkpoint['epoch'] + 1
        i = checkpoint['iteration'] + 1
        num_train_samples = checkpoint['num_train_samples']

    writer = SummaryWriter(
        args.logging_path + args.logging_name + '_' +
        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))



    for epoch in range(chkpnt_epoch, epochs):

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

            for sample in rds.datasets['rnd_train']:
                augmented_batch = []
                batch, augmented = preprocess_image(sample[0][0].squeeze().cpu().detach().numpy(), train=True, mean=mean, std=std)
                augmented_batch.append(augmented)
                for img in sample[0][1:]:
                    input_tensor, augmented_image = preprocess_image(img.squeeze().cpu().detach().numpy(), train=True, mean=mean, std=std)
                    batch = torch.cat((batch,input_tensor), dim=0)
                    augmented_batch.append(augmented_image)
                batch = batch.to(device)

                optimizer.zero_grad()

                labels = sample[2]

                logits_cl, logits_am, heatmap, masked_image, mask = gain(batch, sample[1])

                class_onehot = torch.stack(sample[1]).float()

                cl_loss = loss_fn(logits_cl, class_onehot)

                batch_am_labels_scores = []
                for k in range(len(labels[0])):
                    am_scores = nn.Softmax(dim=1)(logits_am[k])
                    am_labels_scores = am_scores.view(-1)[labels[0][k]]
                    batch_am_labels_scores.append(am_labels_scores)

                num_of_labels = len(sample[2][0])
                am_loss = sum(batch_am_labels_scores) / num_of_labels

                total_loss = num_of_labels * cl_loss * cl_factor + am_loss * am_factor

                epoch_train_am_loss += (am_loss * am_factor).detach().cpu().item()
                epoch_train_cl_loss += (cl_loss * cl_factor).detach().cpu().item()
                epoch_train_total_loss += total_loss.detach().cpu().item()

                writer.add_scalar('Per_Step/train/cl_loss', (cl_loss * cl_factor).detach().cpu().item(), i)
                writer.add_scalar('Per_Step/train/am_loss', (am_loss * am_factor).detach().cpu().item(), i)
                writer.add_scalar('Per_Step/train/total_loss', total_loss.detach().cpu().item(), i)

                loss = num_of_labels * cl_loss * cl_factor
                if gain.AM_enabled():
                    loss += am_loss * am_factor
                loss.backward()
                optimizer.step()

                #Accuracy evaluation
                num_of_labels = len(sample[2][0])
                _, y_pred = logits_cl[0].detach().topk(k=num_of_labels)
                y_pred = y_pred.view(-1).tolist()
                gt = sample[2][0]
                acc = len(set(y_pred).intersection(set(gt))) / num_of_labels
                total_train_single_accuracy += acc

                if i % args.record_itr_train == 0:
                    num_of_labels = len(sample[2][0])
                    for t in range(num_of_labels):
                        one_heatmap = heatmap[t].squeeze().cpu().detach().numpy()

                        one_augmented_im = torch.tensor(np.array(augmented_batch[0])).to(device).unsqueeze(0)
                        one_masked_image = masked_image[t].detach().squeeze()
                        htm = deprocess_image(one_heatmap)
                        visualization, red_htm = show_cam_on_image(one_augmented_im.cpu().detach().numpy(), htm, True)

                        viz = torch.from_numpy(visualization).to(device)
                        masked_im = denorm(one_masked_image, mean, std)
                        masked_im = (masked_im.squeeze().permute([1, 2, 0])
                            .cpu().detach().numpy() * 255).round()\
                            .astype(np.uint8)

                        orig = sample[0][0].unsqueeze(0)
                        masked_im = torch.from_numpy(masked_im).unsqueeze(0).to(device)
                        orig_viz = torch.cat((orig, one_augmented_im, viz, masked_im), 0)
                        grid = torchvision.utils.make_grid(orig_viz.permute([0, 3, 1, 2]))
                        gt = [categories[x] for x in labels[0]]
                        writer.add_image(tag='Train_Heatmaps/image_' + str(i) +
                                             '_' + str(t) + '_'  +'_'.join(gt),
                                         img_tensor=grid, global_step=epoch,
                                         dataformats='CHW')
                        y_scores = nn.Softmax()(logits_cl[0].detach())
                        _, predicted_categories = y_scores.topk(num_of_labels)
                        predicted_cl = [(categories[x], format(y_scores.view(-1)[x], '.4f')) for x in
                                        predicted_categories.view(-1)]
                        labels_cl = [(categories[x], format(y_scores.view(-1)[x], '.4f')) for x in labels[0]]
                        import itertools
                        predicted_cl = list(itertools.chain(*predicted_cl))
                        labels_cl = list(itertools.chain(*labels_cl))
                        cl_text = 'cl_gt_' + '_'.join(labels_cl) + '_pred_' + '_'.join(predicted_cl)
                        am_scores = nn.Softmax(dim=1)(logits_am[t])
                        _, am_labels = am_scores.topk(num_of_labels)
                        predicted_am = [(categories[x], format(am_scores.view(-1)[x], '.4f')) for x in
                                        am_labels.tolist()[0]]
                        labels_am = [(categories[x], format(am_scores.view(-1)[x], '.4f')) for x in labels[0]]
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

        for sample in rds.datasets['seq_test']:

            batch, _ = preprocess_image(sample[0][0].squeeze().cpu().detach().numpy(), train=False, mean=mean, std=std)
            for img in sample[0][1:]:
                input_tensor, input_image = preprocess_image(img.squeeze().cpu().detach().numpy(), train=False,
                                                             mean=mean, std=std)
                batch = torch.cat((batch, input_tensor), dim=0)
            batch = batch.to(device)
            labels = sample[2]

            logits_cl, logits_am, heatmap, masked_image, mask = gain(batch, sample[1])

            num_of_labels = len(sample[2][0])

            # Single label evaluation
            _, y_pred = logits_cl.detach().topk(k=num_of_labels)
            y_pred = y_pred.view(-1).tolist()
            gt = sample[2][0]
            acc = len(set(y_pred).intersection(set(gt))) / num_of_labels
            total_test_single_accuracy += acc

            if j % args.record_itr_test == 0:
                one_heatmap = heatmap[0].squeeze().cpu().detach().numpy()
                one_input_image = sample[0][0].cpu().detach().numpy()
                one_masked_image = masked_image[0].detach().squeeze()
                htm = deprocess_image(one_heatmap)
                visualization, heatmap = show_cam_on_image(one_input_image, htm, True)

                viz = torch.from_numpy(visualization).unsqueeze(0).to(device)
                augmented = torch.tensor(one_input_image).unsqueeze(0).to(device)
                masked_im = denorm(one_masked_image, mean, std)
                masked_im = (masked_im.squeeze().permute([1, 2, 0])
                             .cpu().detach().numpy() * 255).round() \
                    .astype(np.uint8)

                orig = sample[0][0].unsqueeze(0)
                masked_im = torch.from_numpy(masked_im).unsqueeze(0).to(device)
                orig_viz = torch.cat((orig, augmented, viz, masked_im), 0)
                grid = torchvision.utils.make_grid(orig_viz.permute([0, 3, 1, 2]))
                gt = [categories[x] for x in labels[0]]
                writer.add_image(tag='Test_Heatmaps/image_' + str(j) +
                                     '_' + str(0) + '_' + '_'.join(gt),
                                 img_tensor=grid, global_step=epoch,
                                 dataformats='CHW')
                y_scores = nn.Softmax()(logits_cl[0].detach())
                _, predicted_categories = y_scores.topk(num_of_labels)
                predicted_cl = [(categories[x], format(y_scores.view(-1)[x], '.4f')) for x in
                                predicted_categories.view(-1)]
                labels_cl = [(categories[x], format(y_scores.view(-1)[x], '.4f')) for x in labels[0]]
                import itertools
                predicted_cl = list(itertools.chain(*predicted_cl))
                labels_cl = list(itertools.chain(*labels_cl))
                cl_text = 'cl_gt_' + '_'.join(labels_cl) + '_pred_' + '_'.join(predicted_cl)
                am_scores = nn.Softmax(dim=1)(logits_am[0])
                _, am_labels = am_scores.topk(num_of_labels)
                predicted_am = [(categories[x], format(am_scores.view(-1)[x], '.4f')) for x in
                                am_labels.tolist()[0]]
                labels_am = [(categories[x], format(am_scores.view(-1)[x], '.4f')) for x in labels[0]]
                import itertools
                predicted_am = list(itertools.chain(*predicted_am))
                labels_am = list(itertools.chain(*labels_am))
                am_text = '_am_gt_' + '_'.join(labels_am) + '_pred_' + '_'.join(predicted_am)

                writer.add_text('Test_Heatmaps_Description/image_' + str(j) + '_' + str(0) + '_' +
                                '_'.join(gt), cl_text + am_text, global_step=epoch)



            j += 1

        num_test_samples = len(rds.datasets['seq_test'])*batch_size

        if len(args.checkpoint_file_path_save) > 0 and epoch % args.checkpoint_nepoch == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iteration': i,
                'num_train_samples': num_train_samples
            }, args.checkpoint_file_path_save + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        
        print("finished epoch number:")
        print(epoch)


        if (test_first and epoch > 0) or test_first == False:
            writer.add_scalar('Loss/train/cl_total_loss', epoch_train_cl_loss / (num_train_samples*batch_size), epoch)
            writer.add_scalar('Loss/train/am_tota_loss', epoch_train_am_loss / num_train_samples, epoch)
            writer.add_scalar('Loss/train/combined_total_loss', epoch_train_total_loss / num_train_samples, epoch)
            writer.add_scalar('Accuracy/train/cl_accuracy', total_train_single_accuracy / (num_train_samples*batch_size), epoch)


        writer.add_scalar('Accuracy/test/cl_accuracy', total_test_single_accuracy / num_test_samples, epoch)
        
        gain.increase_epoch_count()




if __name__ == '__main__':
    args = parser.parse_args()
    main(args)