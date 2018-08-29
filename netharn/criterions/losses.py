# source: https://github.com/adambielski/siamese-triplet/blob/master/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - distances.sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    """

    def __init__(self, margin, pair_selector):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
      #  print('embeddings losses', embeddings)
      #  print('target losses', target)
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
       # print('pos pairs',positive_pairs)
       # print('neg pairs', negative_pairs)
       
        try:
            num = len(negative_pairs)*2 
        except:
            num = 1
        #positive_pairs = torch.from_numpy(positive_pairs).cuda() #torch.tensor(positive_pairs)
        negative_pairs = torch.from_numpy(np.array(negative_pairs, dtype='float32')) #torch.tensor(negative_pairs)
        #print('pos pairs',positive_pairs)
        #print('neg pairs', negative_pairs)
        hinge_neg_dist = torch.clamp(self.margin - negative_pairs, min=0.0)
        loss_imposter = torch.pow(hinge_neg_dist, 2)
        loss_genuine = torch.pow(positive_pairs, 2)
        loss2x = loss_genuine + loss_imposter
        #ave_loss = loss2x.mean()
        ave_loss = torch.sum(loss2x) / 2.0 / num
        loss = ave_loss
        return loss

class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)
