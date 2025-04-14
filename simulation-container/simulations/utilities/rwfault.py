from utilities.fault import Fault
from collections.abc import Collection
from collections import OrderedDict
from typing import Any
import random
from Basilisk import messaging
import numpy as np

class RWFault(Fault):
    def __init__(self, components: Collection[Any], rwFactory, rwEffector, **kwargs):
        super().__init__(components, **kwargs)
        self.rwFactory = rwFactory
        self.rwEffector = rwEffector
        self.rwModel = kwargs.get("model")
        self.seed = kwargs.get("seed")
    
    def replaceRWs(self, newRWs):
        self.rwFactory.rwList = OrderedDict([])
        self.rwEffector.ReactionWheelData = self.rwEffector.ReactionWheelData[:0]
        for (rwType, spin, torque) in newRWs:
            rwModel = self.rwModel if self.rwModel else messaging.BalancedWheels
            RW_new = self.rwFactory.create(rwType, spin, maxMomentum=100., RWmodel=rwModel, u_max=torque)
            self.rwEffector.addReactionWheel(RW_new)

    def reset(self):
        if not self.defaults:
            raise ValueError("Defaults have not been set.")
        resetRWs = []
        try:
            for i in range(len(self.components)):
                spin = self.defaults[i]["axis"]
                u_max = self.defaults[i]["u_max"]
                rwType = self.defaults[i]["type"]
                resetRWs.append((rwType, spin, u_max))
        except:
            raise TypeError("Defaults and/or Components do not contain necessary information.")
        self.replaceRWs(resetRWs)
        return len(self.rwFactory.rwList) > 0 and len(self.rwEffector.ReactionWheelData) > 0

    def inject(self, newSettings) -> Collection[Any]:
        self.replaceRWs(newSettings)
        return newSettings

    def inject_random(self, chance : float) -> Collection[tuple[bool, Any]]:
        toInject = self.randomize(chance, seed=self.seed)
        if self.seed:
            random.seed(self.seed)
        newSettings = []
        for i in range(len(toInject)):
            spin = [sublist[0] for sublist in self.components[i][1].gsHat_B]
            rwType = self.components[i][0]
            u_max = self.components[i][1].u_max
            if toInject[i]:
                rand = random.random() * 10
                u_max = u_max / rand if rand >= 1 else u_max * rand
            newSettings.append((rwType, spin, u_max))
        details = list(zip(toInject, self.inject(newSettings)))
        for i in range(len(details)):
            if not details[i][0]:
                details[i] = (details[i][0], None)
        return np.array(details, dtype=object)
        