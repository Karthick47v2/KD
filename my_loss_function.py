import torch.nn as nn
import torch.nn.functional as F


def loss_kd(outputs, labels, teacher_outputs, params):
    """
    loss function for Knowledge Distillation (KD)
    """
    alpha = params.alpha
    T = params.temperature

    loss_CE = F.cross_entropy(outputs, labels)
    D_KL = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                          F.softmax(teacher_outputs/T, dim=1)) * (T * T)
    KD_loss = (1. - alpha)*loss_CE + alpha*D_KL

    return KD_loss
