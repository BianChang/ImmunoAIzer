# This is the source code for ImmunoAIzer: biomarker prediction network
# We have implemented four networks in this code including:u-net, deeplabv3,deeplabv3+, and the proposed network based on Inception V3
# @Author: Chang Bian
# date: 2021-01-11

import logging
import argparse
import torch
import torch.nn as nn
#from torch.utils import data, model_zoo
import numpy as np
import random

from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import torch.backends.cudnn as cudnn

import os


from eval import eval_net
from modified_unet import modified_UNet
from unet import UNet
from models.deeplabv3_plus import DeepLabV3Plus
from models.deeplabv3 import DeepLabV3
from models.nestedunet import NestedUNet
from models.inception import Inception3
from model.discriminator import FCDiscriminator
from utils.loss import CrossEntropy2d, BCEWithLogitsLoss2d

from utils.dataset import BasicDataset, UnlabeledDataset
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import random

MODEL = 'modified_UNet'
BATCH_SIZE = 2
ITER_SIZE = 1
IMG_DIRECTORY = './data/imgs/'
MASK_DIRECTORY = './data/masks/'
TCGA_DIRECTORY = './data/tcga/'
DIR_CHECKPOINTS = 'checkpoints/'
IGNORE_LABEL = 255
INPUT_SIZE = '512,512'
LEARNING_RATE = 0.001
MOMENTUM = 0.9
NUM_CLASSES = 4
NUM_STEPS = 300000
POWER = 0.9
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 5000
SVAE_CP = True
WEIGHT_DECAY = 1e-8

LEARNING_RATE_D = 1e-6
LAMBDA_ADV_PRED = 0.01

SEMI_TRAIN = True

SEMI_START=100000
LAMBDA_SEMI=0.1
MASK_T=0.5

LAMBDA_SEMI_ADV=0.001
SEMI_START_ADV=10000
D_REMAIN=True


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="modified-Unet Network")
    parser.add_argument('-m', "--model", dest='mod', type=str, default=MODEL,
                        help="available options : DeepLab/DRN")
    parser.add_argument('-b', "--batch-size", dest='batch_size', type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--semi-train", type=bool, dest='semi_train', default=SEMI_TRAIN,
                        help="choose semi train.")
    parser.add_argument("--partial-id", type=str, default=None,
                        help="restore partial id list")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument('-l', "--learning-rate", type=float, dest='lr', default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument('-ld',"--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-adv-pred", type=float, default=LAMBDA_ADV_PRED,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-semi", type=float, default=LAMBDA_SEMI,
                        help="lambda_semi for adversarial training.")
    parser.add_argument("--lambda-semi-adv", type=float, default=LAMBDA_SEMI_ADV,
                        help="lambda_semi for adversarial training.")
    parser.add_argument("--mask-T", type=float, default=MASK_T,
                        help="mask T for semi adversarial training.")
    parser.add_argument("--save-cp", type=bool, dest='save_cp', default=SVAE_CP,
                        help="save checkpoints.")
    parser.add_argument("--semi-start", type=int, default=SEMI_START,
                        help="start semi learning after # iterations")
    parser.add_argument("--semi-start-adv", type=int, default=SEMI_START_ADV,
                        help="start semi learning after # iterations")
    parser.add_argument("--D-remain", type=bool, default=D_REMAIN,
                        help="Whether to train D with unlabeled data")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, dest='num_steps', default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--restore-from-D", type=str, default=None,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1.0,
                        help='Downscaling factor of the images')
    return parser.parse_args()

args = get_arguments()

def loss_calc(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda()
    criterion = CrossEntropy2d().cuda()

    return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.lr, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def one_hot(label):
    one_hot = torch.zeros((label.shape[0], args.num_classes, label.shape[2], label.shape[3]), dtype=label.dtype)
    one_hot.scatter_(1,label.long(),1)
    one_hot = one_hot.float()
    """
    label = label.numpy()
    one_hot = np.zeros((label.shape[0], args.num_classes, label.shape[2], label.shape[3]), dtype=label.dtype)
    print(one_hot.shape)
    print(label.shape)
    for i in range(args.num_classes):
        one_hot[:, i, :, :] = (label == i)
    """
    #handle ignore labels
    return torch.FloatTensor(one_hot)

def make_D_label(label, ignore_mask):
    ignore_mask = np.expand_dims(ignore_mask, axis=1)
    D_label = np.ones(ignore_mask.shape)*label
    D_label[ignore_mask] = 255
    D_label = Variable(torch.FloatTensor(D_label)).cuda()

    return D_label


def main():

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    cudnn.enabled = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # initialize parameters
    num_steps = args.num_steps
    batch_size = args.batch_size
    lr = args.lr
    save_cp = args.save_cp
    img_scale = args.scale
    val_percent = args.val / 100

    # data input
    dataset = BasicDataset(IMG_DIRECTORY, MASK_DIRECTORY, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    tcga_dataset = UnlabeledDataset(TCGA_DIRECTORY)
    n_unlabeled = len(tcga_dataset)

    # create network
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    #logger.addHandler(logging.StreamHandler())
    logging.info('Using device %s' % str(device))
    logging.info('Network %s' % args.mod)
    logging.info('''Starting training:
            Num_steps:          %.2f
            Batch size:      %.2f
            Learning rate:   %.4f_transform
            Training size:   %.0f
            Validation size: %.0f
            Unlabeled size:  %.0f
            Checkpoints:     %s
            Device:          %s
            Scale:           %.2f
        ''' % (num_steps, batch_size, lr, n_train, n_val, n_unlabeled, str(save_cp), str(device.type), img_scale))
    if args.mod == 'unet':
        net = UNet(n_channels=3, n_classes=NUM_CLASSES)
        print('channels = %d , classes = %d' % (net.n_channels, net.n_classes))
    elif args.mod == 'modified_unet':
        net = modified_UNet(n_channels=3, n_classes=NUM_CLASSES)
        print('channels = %d , classes = %d' % (net.n_channels, net.n_classes))
    elif args.mod == 'deeplabv3':
        net = DeepLabV3(nclass=NUM_CLASSES, pretrained_base=False)
        print('channels = 3 , classes = %d' % net.nclass)
    elif args.mod == 'deeplabv3plus':
        net = DeepLabV3Plus(nclass=NUM_CLASSES, pretrained_base=False)
        print('channels = 3 , classes = %d' % net.nclass)
    elif args.mod == 'nestedunet':
        net = NestedUNet(nclass=NUM_CLASSES, deep_supervision=False)
        print('channels = 3 , classes = %d' % net.nlass)
    elif args.mod == 'inception3':
        net = Inception3(n_classes=4, inception_blocks=None, init_weights=True, bilinear=True)
        print('channels = 3 , classes = %d' % net.n_classes)

    net.to(device=device)
    net.train()

    cudnn.benchmark = True

    # init D
    model_D = FCDiscriminator(num_classes=args.num_classes)
    if args.restore_from_D is not None:
        model_D.load_state_dict(torch.load(args.restore_from_D))
    model_D.train()
    model_D.cuda()


    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    if args.semi_train is None:
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    else:
        #read unlabeled data and labeled data
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        trainloader_remain = DataLoader(tcga_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        #trainloader_gt = data.DataLoader(train_gt_dataset,
                        #batch_size=args.batch_size, sampler=train_gt_sampler, num_workers=3, pin_memory=True)

        trainloader_remain_iter = enumerate(trainloader_remain)


    trainloader_iter = enumerate(train_loader)


    # implement model.optim_parameters(args) to handle different models' lr setting

    # optimizer for segmentation network
    #optimizer = optim.SGD(net.optim_parameters(args),
                #lr=args.learning_rate, momentum=args.momentum,weight_decay=args.weight_decay)
    optimizer = optim.Adam(net.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    optimizer.zero_grad()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10000, eta_min=1e-6, last_epoch=-1)

    # optimizer for discriminator network
    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9,0.99))
    #optimizer_D = optim.SGD(model_D.parameters(), lr=args.learning_rate_D, momentum=args.momentum,weight_decay=args.weight_decay)
    optimizer_D.zero_grad()

    # loss/ bilinear upsampling
    bce_loss = BCEWithLogitsLoss2d()
    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)

    '''
    if version.parse(torch.__version__) >= version.parse('0.4.0'):
        interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)
    else:
        interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')
    '''

    # labels for adversarial training
    pred_label = 0
    gt_label = 1


    for i_iter in range(args.num_steps):

        best_acc = 0
        loss_seg_value = 0
        loss_adv_pred_value = 0
        loss_D_value = 0
        loss_semi_value = 0
        loss_semi_adv_value = 0

        optimizer.zero_grad()
        #adjust_learning_rate(optimizer, i_iter)
        optimizer_D.zero_grad()
        adjust_learning_rate_D(optimizer_D, i_iter)

        for sub_i in range(args.iter_size):

            # train G

            # don't accumulate grads in D
            for param in model_D.parameters():
                param.requires_grad = False
            for param in net.parameters():
                param.requires_grad = True


            # do semi first
            if (args.lambda_semi > 0 or args.lambda_semi_adv > 0 ) and i_iter >= args.semi_start_adv :
                try:
                    _, batch = trainloader_remain_iter.__next__()
                except:
                    trainloader_remain_iter = enumerate(trainloader_remain)
                    _, batch = trainloader_remain_iter.__next__()

                # only access to img
                images = batch['image']
                images = images.type(torch.FloatTensor)
                images = Variable(images).cuda()

                pred = net(images)
                pred_remain = pred.detach()



                D_out = interp(model_D(F.softmax(pred, dim=1)))
                D_out_sigmoid = torch.sigmoid(D_out).data.cpu().numpy().squeeze(axis=1)
                #D_out_sigmoid = torch.sigmoid(D_out).data.cpu().numpy()

                #ignore_mask_remain = np.zeros(D_out_sigmoid.shape).astype(np.bool)

                targetr = Variable(torch.ones(D_out.shape))
                targetr = Variable(torch.FloatTensor(targetr)).cuda()
                loss_semi_adv = args.lambda_semi_adv * bce_loss(D_out, targetr)
                loss_semi_adv = loss_semi_adv/args.iter_size

                #loss_semi_adv.backward()
                #loss_semi_adv_value += loss_semi_adv.data.cpu().numpy()[0]/args.lambda_semi_adv
                loss_semi_adv_value += loss_semi_adv.cpu().detach().numpy().item() / args.lambda_semi_adv

                if args.lambda_semi <= 0 or i_iter < args.semi_start:
                    loss_semi_adv.backward()
                    loss_semi_value = 0
                else:
                    # produce ignore mask
                    semi_ignore_mask = (D_out_sigmoid < args.mask_T)

                    semi_gt = pred.data.cpu().numpy().argmax(axis=1)
                    semi_gt[semi_ignore_mask] = 255

                    semi_ratio = 1.0 - float(semi_ignore_mask.sum())/semi_ignore_mask.size
                    print('semi ratio: {:.4f}'.format(semi_ratio))

                    if semi_ratio == 0.0:
                        loss_semi_value += 0
                    else:
                        semi_gt = torch.FloatTensor(semi_gt)

                        loss_semi = args.lambda_semi * loss_calc(pred, semi_gt)
                        loss_semi = loss_semi/args.iter_size
                        loss_semi_value += loss_semi.cpu().detach().numpy().item()/args.lambda_semi
                        loss_semi += loss_semi_adv
                        loss_semi.backward()

            else:
                loss_semi = None
                loss_semi_adv = None

            # train with source

            try:
                _, batch = trainloader_iter.__next__()
            except:
                trainloader_iter = enumerate(train_loader)
                _, batch = trainloader_iter.__next__()

            images = batch['image']
            labels = batch['mask']
            images = images.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)
            labels = labels.squeeze(1)
            ignore_mask = (labels.cpu().numpy() == 255)
            #pred = interp(net(images))

            pred = net(images)
            criterion = nn.CrossEntropyLoss()
            loss_seg = criterion(pred, labels)
            #loss_seg = loss_calc(pred, labels)

            D_out = interp(model_D(F.softmax(pred, dim=1)))

            targetr = Variable(torch.ones(D_out.shape))
            targetr = Variable(torch.FloatTensor(targetr)).cuda()
            #loss_adv_pred = bce_loss(D_out, targetr)

            if i_iter > args.semi_start_adv:
                loss_adv_pred = bce_loss(D_out, targetr)
                loss = loss_seg + args.lambda_adv_pred * loss_adv_pred
                loss_adv_pred_value += loss_adv_pred.cpu().detach().numpy().item() / args.iter_size
            else:
                loss = loss_seg

            # proper normalization
            loss = loss/args.iter_size
            loss.backward()
            optimizer.step()
            loss_seg_value += loss_seg.cpu().detach().numpy().item()/args.iter_size
            #loss_adv_pred_value += loss_adv_pred.cpu().detach().numpy().item()/args.iter_size


            # train D

            # bring back requires_grad
            if i_iter > args.semi_start_adv and i_iter % 3 == 0:
                for param in net.parameters():
                    param.requires_grad = False
                for param in model_D.parameters():
                    param.requires_grad = True

            # train with pred
                pred = pred.detach()

                if args.D_remain:
                    pred = torch.cat((pred, pred_remain), 0)
                #ignore_mask = np.concatenate((ignore_mask,ignore_mask_remain), axis = 0)

                D_out = interp(model_D(F.softmax(pred, dim=1)))
            #targetf = Variable(torch.zeros(D_out.shape))
                targetf = 0.1 * np.random.rand(D_out.shape[0], D_out.shape[1], D_out.shape[2], D_out.shape[3])
                targetf = Variable(torch.FloatTensor(targetf)).cuda()
                loss_D = bce_loss(D_out, targetf)
                loss_D = loss_D/args.iter_size/2
                loss_D.backward()
                loss_D_value += loss_D.data.cpu().detach().numpy().item()


            # train with gt
            # get gt labels
                try:
                    _, batch = trainloader_iter.__next__()
                except:
                    trainloader_iter = enumerate(train_loader)
                    _, batch = trainloader_iter.__next__()

                labels_gt = batch['mask']
                D_gt_v = Variable(one_hot(labels_gt)).cuda()
                ignore_mask_gt = (labels_gt.numpy() == 255).squeeze(axis=1)

                D_out = interp(model_D(D_gt_v))
                #targetr = Variable(torch.ones(D_out.shape))
                targetr = 0.1 * np.random.rand(D_out.shape[0], D_out.shape[1], D_out.shape[2], D_out.shape[3]) + 0.9
                targetr = Variable(torch.FloatTensor(targetr)).cuda()
                loss_D = bce_loss(D_out, targetr)
                loss_D = loss_D/args.iter_size/2
                loss_D.backward()
                optimizer_D.step()
                loss_D_value += loss_D.cpu().detach().numpy().item()
        scheduler.step()


        print('iter = {0:8d}/{1:8d}, loss_seg = {2:.3f}, loss_adv_p = {3:.3f}, loss_D = {4:.3f}, loss_semi = {5:.3f}, loss_semi_adv = {6:.3f}'.format(i_iter, args.num_steps, loss_seg_value, loss_adv_pred_value, loss_D_value, loss_semi_value, loss_semi_adv_value))

        '''
        if i_iter >= args.num_steps-1:
            print 'save model ...'
            torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'VOC_'+str(args.num_steps)+'.pth'))
            torch.save(model_D.state_dict(),osp.join(args.snapshot_dir, 'VOC_'+str(args.num_steps)+'_D.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter!=0:
            print 'taking snapshot ...'
            torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'VOC_'+str(i_iter)+'.pth'))
            torch.save(model_D.state_dict(),osp.join(args.snapshot_dir, 'VOC_'+str(i_iter)+'_D.pth'))
        '''
        # save checkpoints
        if save_cp and (i_iter % 1000) == 0 and (i_iter != 0):
            try:
                os.mkdir(DIR_CHECKPOINTS)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       DIR_CHECKPOINTS + 'i_iter_%d.pth' % (i_iter + 1))
            logging.info('Checkpoint %d saved !' % (i_iter + 1))

        if (i_iter % 1000 == 0) and (i_iter != 0):
            val_score, accuracy, dice_avr, dice_panck, dice_nuclei, dice_lcell = eval_net(net, val_loader, device, n_val)
            logging.info('Validation cross entropy: {}'.format(val_score))
            if accuracy > best_acc:
                best_acc = accuracy
            result_file = open('result.txt', 'a', encoding='utf-8')
            result_file.write('best_acc = ' + str(best_acc) + '\n' + 'iter = ' + str(i_iter) + '\n')
            result_file.close




if __name__ == '__main__':
    main()
