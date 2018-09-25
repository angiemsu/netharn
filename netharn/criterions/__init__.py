# -*- coding: utf-8 -*-
"""
mkinit netharn.criterions
"""
# flake8: noqa
from __future__ import absolute_import, division, print_function, unicode_literals
from torch.nn.modules.loss import CrossEntropyLoss, MSELoss, TripletMarginLoss

__DYNAMIC__ = False
if __DYNAMIC__:
    from mkinit import dynamic_init
    exec(dynamic_init(__name__))
else:
   # from netharn.criterions import contrastive_loss
   # from netharn.criterions import focal
   # from netharn.criterions.contrastive_loss import (ContrastiveLoss,)
 #   from netharn.criterions.focal import (FocalLoss, one_hot_embedding,)
  #  from netharn.criterions import losses
   # from netharn.criterions.losses import (OnlineContrastiveLoss)
    # from . import contrastive_loss
    from . import focal
    from .contrastive_loss import (ContrastiveLoss,)
    from .focal import (FocalLoss, one_hot_embedding,)

    __all__ = ['ContrastiveLoss', 'FocalLoss', 'contrastive_loss', 'focal',
               'one_hot_embedding']
