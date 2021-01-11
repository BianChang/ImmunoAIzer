import argparse
import logging
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import imageio

from unet import UNet
from models.deeplabv3_plus import DeepLabV3Plus
from models.deeplabv3 import DeepLabV3
from models.nestedunet import NestedUNet
from models.inception import Inception3
from modified_unet import modified_UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset
#from utils.crf import dense_crf
from postprocessing import nuclei_process
import scipy.io as io
from Metrics import accuracy_score, precision_score, recall_score, f1_score, IOU
from Metrics import diceCoeff_avr
from Metrics import diceCoeff_panck
from Metrics import diceCoeff_nuclei
from Metrics import diceCoeff_lcell
import configparser


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                use_dense_crf=False):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        #output,aux = net(img)
        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.shape[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    if use_dense_crf:
        full_mask = dense_crf(np.array(full_img).astype(np.uint8), full_mask)

    # return full_mask > out_threshold
    return np.argmax(full_mask, axis=0)


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('-n', '--network', dest='net', type=str, default='unet',
                        help='Load Network')
    return parser.parse_args()


def mask_to_image(mask):
    return Image.fromarray((mask * 50).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()
    config = configparser.ConfigParser()
    config.read('config.ini', encoding='utf-8')
    in_files = config.get('folder', 'input')
    true_masks = config.get('folder', 'truemasks')
    output = config.get('folder', 'output')

    if args.net == 'unet':
        net = UNet(n_channels=3, n_classes=4)
        print('channels = %d , classes = %d' % (net.n_channels, net.n_classes))
    elif args.net == 'modified_unet':
        net = modified_UNet(n_channels=3, n_classes=4)
        print('channels = %d , classes = %d' % (net.n_channels, net.n_classes))
    elif args.net == 'deeplabv3':
        net = DeepLabV3(nclass=4, pretrained_base=False)
        print('channels = 3 , classes = %d' % net.nclass)
    elif args.net == 'deeplabv3plus':
        net = DeepLabV3Plus(nclass=4, pretrained_base=False)
        print('channels = 3 , classes = %d' % net.nclass)
    elif args.net == 'nestedunet':
        net = NestedUNet(nclass=4, deep_supervision=False)
        print('channels = 3 , classes = %d' % net.nlass)
    elif args.net == 'inception3':
        net = Inception3(n_classes=4, inception_blocks=None, init_weights=True, bilinear=True)
        print('channels = 3 , classes = %d' % net.n_classes)
    #net = UNet(n_channels=3, n_classes=4)

    logging.info("Loading model {}".format(args.model))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    logging.info("Model loaded !")
    accuracy = 0
    precision = 0
    recall = 0
    f1 = 0
    iou = 0
    diceCoeff = 0
    count = 0
    dice_panck = 0
    dice_nuclei = 0
    dice_lcell = 0

    for filename in os.listdir(in_files):
        count = count + 1
        img = Image.open(in_files + "/" + filename)
        img_array = np.array(img)
        img = img_array
        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=1.0,
                           use_dense_crf=False,
                           device=device)
        mask = nuclei_process(mask)
        true_mask = Image.open(true_masks + "/" + filename[:-4] + ".bmp")
        true_mask = np.array(true_mask)
        accuracy += accuracy_score(mask, true_mask)
        precision += precision_score(mask, true_mask)
        recall += recall_score(mask, true_mask)
        f1 += f1_score(mask, true_mask)
        iou += IOU(mask, true_mask)
        diceCoeff += diceCoeff_avr(mask, true_mask)
        dice_panck += diceCoeff_panck(mask, true_mask)
        dice_nuclei += diceCoeff_nuclei(mask, true_mask)
        dice_lcell += diceCoeff_lcell(mask, true_mask)
        mask_viz = mask_to_image(mask)
        #imageio.imwrite(output + "/" + filename, mask_viz)
        mask = Image.fromarray((mask).astype(np.uint8))
        imageio.imwrite(output + "/" + filename[:-4]+'.bmp', mask)

    print("num of samples%.0f" % count)
    print("accuracy = %.3f" % (accuracy / count))
    print("precision = %.3f" % (precision / count))
    print("recall = %.3f" % (recall / count))
    print("F1 score = %.3f" % (f1 / count))
    print("IOU = %.3f" % (iou / count))
    print("average dice = %.3f" % (diceCoeff / count))
    print("panCK dice = %.3f" % (dice_panck / count))
    print("nuclei dice = % .3f" % (dice_nuclei / count))
    print("TILS dice = %.3f" % (dice_lcell / count))