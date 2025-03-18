from faults.fault import Fault
from collections.abc import Collection
from collections import OrderedDict
from typing import Any
from Basilisk import messaging

class RWFault(Fault):
    def __init__(self, components: Collection[Any], rwFactory, rwEffector, **kwargs):
        super().__init__(components, **kwargs)
        self.rwFactory = rwFactory
        self.rwEffector = rwEffector
        self.rwModel = kwargs.get("model") if kwargs.get("model") else None
    
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

    def inject(self) -> Collection[Any]:
        pass

    def inject_random(self, chance : float) -> Collection[tuple[bool, Any]]:
        pass
        