from __future__ import absolute_import

from .oim import oim, OIM, OIMLoss
from .triplet import TripletLoss
from .deep_loss import DeepLoss

__all__ = [
    'oim',
    'OIM',
    'OIMLoss',
    'TripletLoss',
    'DeepLoss',
]
