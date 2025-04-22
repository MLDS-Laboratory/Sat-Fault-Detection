from utilities.fault import Fault
from collections.abc import Collection
from collections import OrderedDict
from typing import Any
import random
from Basilisk.architecture import sysModel, messaging
from Basilisk.simulation import coarseSunSensor
import numpy as np

class CSSfault(sysModel.SysModel):
    def __init__(self, components : Collection[Any], **kwargs):
        defaults = kwargs.get("defaults")
        self.components = components
        self.defaults = defaults
        self.types = ["OFF", "STUCK_CURRENT", "STUCK_MAX", "STUCK_RAND", "RAND", "NOMINAL"]
        super().__init__()
        self.faultState = ["NOMINAL"] * len(self.components)
        self.chance = kwargs.get("chance")
        self.seed = kwargs.get("seed")
        for i in self.components:
            i.faultState = getattr(coarseSunSensor, "NOMINAL")
        
    def UpdateState(self, CurrentSimNanos):
        self.faultState = [i[1] for i in self.inject_random(self.chance)]
    def inject(self, newSettings : list):
        for i in range(len(self.components)):           
            try:
                if newSettings[i].upper() == "NOMINAL":
                    self.components[i].faultState = getattr(coarseSunSensor, (newSettings[i]).upper())
                else:
                    self.components[i].faultState = getattr(coarseSunSensor, "CSSFAULT_" + (newSettings[i]).upper())
            except:
                raise NameError(newSettings[i] + " is not yet a supported fault type.")

        return newSettings

    def inject_random(self, chance):
        if not chance:
            chance = 0.0005
        toInject = self.randomize(chance, seed=self.seed)
        newSettings = []
        for i in range(len(toInject)):
            if toInject[i]:
                fault = np.random.choice(self.types, 1)[0]
                newSettings.append(fault)
            else:
                past = self.components[i].faultState
                if past == 4:
                    newSettings.append("NOMINAL")
                else:
                    newSettings.append(self.types[self.components[i].faultState])
        return list(zip(toInject, self.inject(newSettings)))
    
    def reset(self):
        if not self.defaults:
            raise ValueError("Defaults have not been set.")
        for i in range(len(self.components)):
            try:
                self.components[i].faultState = getattr(coarseSunSensor, "CSSFAULT_" + (self.defaults[i]).upper())
            except:
                raise TypeError("Defaults and/or Components do not contain necessary information.")
    
    def randomize(self, chance : float, **kwargs):
        """
        Choose which components to randomly have faults injected. 

        Parameter:
            chance (float): the chance that each component will fault.

        Returns:
            toInject (list[bool]): Indicates whether a fault is to be injected to the component at this index 
        """
        if not 0. <= chance <= 1.:
            raise ValueError("Chance must be between 0 and 1.")
        seed = kwargs.get("seed")
        if seed is not None:
            random.seed(seed)
        toInject = [random.random() < chance for _ in self.components]
        return toInject
    