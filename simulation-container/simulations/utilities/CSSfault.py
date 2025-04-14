from fault import Fault
from collections.abc import Collection
from collections import OrderedDict
from typing import Any
import random
from Basilisk.architecture import sysModel, messaging
import numpy as np

class CSSfault(sysModel.SysModel):
    def __init__(self, components : Collection[Any], **kwargs):
        defaults = kwargs.get("defaults")
        self.components = components
        self.fault = Fault(components, defaults)
        sysModel.SysModel.__init__()
        

    def inject(self, newSettings : list[dict]):
        for i in range(len(self.components)):
            eval("self." + newSettings["type"])
        return newSettings

    def inject_random(self, chance):
        return super().inject_random(chance)
    
    def reset(self):
        return super().reset()
    