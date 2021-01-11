import torch
import torch.nn.functional as F
import numpy as np
from dice_loss import dice_coeff
import scipy.io as io
import torchvision

from PIL import Image
#from torch.utils.tensorboard import SummaryWriter
from Metrics import accuracy_score
from Metrics import diceCoeff_avr
from Metrics import diceCoeff_panck
from Metrics import diceCoeff_nuclei
from Metrics import diceCoeff_lcell
from postprocessing import nuclei_process

data_transform = torchvision.transforms.Compose([
          #  torchvision.transforms.Resize((128,128)),
         #   torchvision.transforms.CenterCrop(96),
            torchvision.transforms.ToTensor(),
            #torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])


def eval_net(net, loader, device, n_val):
    """Evaluation without the densecrf with the dice coefficient"""
    print('eval in process')
    net.eval()
    tot = 0
    accuracy = 0
    dice_panck = 0
    dice_nuclei = 0
    dice_lcell = 0
    dice_avr = 0
    for batch in loader:
        imgs = batch['image']
        true_masks = batch['mask']

        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=torch.long)
        true_masks = true_masks.squeeze(1)
        # mask_pred, aux = net(imgs) #for deeplabv3 and deeplabv3+
        mask_pred = net(imgs)
        for true_mask, pred in zip(true_masks, mask_pred):

            if net.n_classes > 1:
                tot += F.cross_entropy(pred.unsqueeze(dim=0), true_mask.unsqueeze(dim=0)).item()
            else:
                tot += dice_coeff(pred, true_mask.squeeze(dim=1)).item()
                    
            if net.n_classes > 1:
                probs = F.softmax(pred, dim=1)
            else:
                probs = torch.sigmoid(pred)

            probs = probs.cpu().detach().numpy()

            true_mask1 = true_mask.cpu().numpy()

            probs = np.argmax(probs, axis=0)
            #probs = nuclei_process(probs)
            accuracy += accuracy_score(probs, true_mask1)
            dice_avr += diceCoeff_avr(probs, true_mask1)
            dice_panck += diceCoeff_panck(probs, true_mask1)
            dice_nuclei += diceCoeff_nuclei(probs, true_mask1)
            dice_lcell += diceCoeff_lcell(probs, true_mask1)
    return tot / n_val, accuracy / n_val, dice_avr / n_val, dice_panck / n_val, dice_nuclei / n_val, dice_lcell / n_val
