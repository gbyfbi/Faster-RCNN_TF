# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from .VGGnet_train import VGGnet_train
from .VGGnet_test import VGGnet_test
from .Alexnet_train import Alexnet_train
from .Alexnet_test import Alexnet_test
from .Alexnet_test_debug import Alexnet_test_debug
from . import factory
