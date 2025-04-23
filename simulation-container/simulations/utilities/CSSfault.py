from utilities.fault import Fault
from collections.abc import Collection
from collections import OrderedDict
from typing import Any
import random
from Basilisk.architecture import sysModel, messaging
from Basilisk.simulation import coarseSunSensor
import numpy as np
"""
This is a module for introducing faults into Coarse Sun Sensors. It is a Basilisk Module, so it will
run at every time step provided it is added to a simulation task. 

The module is essentially a way to randomize the built-in faulted states of the coarseSunSensor module. 
It randomly assigns random faults to each CSS. If the randomly chosen fault is anything except NOMINAL
and RAND, that sensor is faulted until it gets a different randomly chosen fault. If the randomly chosen fault is NOMINAL, 
that sensor is not faulted! If the randomly chosen fault is RAND, that sensor returns to NOMINAL the next turn, 
and does not have an entry in the fault sweepstakes that turn.  
"""
class CSSfault(sysModel.SysModel):
    """
    Requires:
    - A list of the sensors
    Optional:
    - A list of default fault states and other things for the sensors
    - a chance for the random fault injections
    - a seed for the random fault injections
    """
    def __init__(self, components : Collection[Any], **kwargs):
        defaults = kwargs.get("defaults")
        self.components = components
        self.defaults = defaults
        self.types = ["OFF", "STUCK_CURRENT", "STUCK_MAX", "STUCK_RAND", "RAND", "NOMINAL"]
        super().__init__()
        #loggable attribute for fault states
        self.faultState = ["NOMINAL"] * len(self.components) #all sensors are assumed to start off nominal
        self.chance = kwargs.get("chance")
        self.seed = kwargs.get("seed")
        for i in self.components:
            i.faultState = getattr(coarseSunSensor, "NOMINAL") #forcibly make all sensors nominal at first
    """
    Update method. Runs at every timestep. 

    Just calls the random fault injection and uses its results to update the loggable attribute. 
    """
    def UpdateState(self, CurrentSimNanos):
        self.faultState = [i[1] for i in self.inject_random(self.chance)]
    
    """
    Injection if specific new settings are desired. 

    Goes through the list of sensors and sets them to the desired fault state. 
    Used by inject_random once new settings have been randomized. Returns the
    new settings. 
    """
    def inject(self, newSettings : list):
        for i in range(len(self.components)):           
            try:
                #for some reason, all the predefined fault states are CSSFAULT_something except NOMINAL
                if newSettings[i].upper() == "NOMINAL":
                    self.components[i].faultState = getattr(coarseSunSensor, (newSettings[i]).upper())
                else:
                    self.components[i].faultState = getattr(coarseSunSensor, "CSSFAULT_" + (newSettings[i]).upper())
            except:
                raise NameError(newSettings[i] + " is not yet a supported fault type.")
        return newSettings
    """
    Random injection of faults with a given chance. 

    Whether each sensor is faulted is randomized, and the type of fault is also
    randomized. As stated above, if the fault chosen is RAND, it will only last 1 turn, 
    and return to NOMINAL the next. Otherwise, it will remain at the chosen or prior state 
    until changed here. This returns a list of the following form:
    (bool, fault state) where bool is whether a fault is injected into the given component
    and fault state is the new (or old, if no new fault) setting. 
    """
    def inject_random(self, chance):
        if not chance:
            chance = 0.0005
        toInject = self.randomize(chance, seed=self.seed)
        newSettings = []
        for i in range(len(toInject)):
            if toInject[i]:
                #randomly choose the fault type
                fault = np.random.choice(self.types, 1)[0]
                newSettings.append(fault)
            else:
                #check if the last fault state was RAND, and change back to NOMINAL if so
                past = self.components[i].faultState
                if past == 4:
                    newSettings.append("NOMINAL")
                else:
                    newSettings.append(self.types[self.components[i].faultState])
        return list(zip(toInject, self.inject(newSettings)))
    
    """
    Resets all reaction wheels to the prescribed defaults, if prescribed. 

    Will eventually need to be updated to reset to NOMINAL if not prescribed, 
    but doesn't really matter because this is not used in normal operation. 
    """
    def reset(self):
        if not self.defaults:
            raise ValueError("Defaults have not been set.")
        for i in range(len(self.components)):
            try:
                self.components[i].faultState = getattr(coarseSunSensor, "CSSFAULT_" + (self.defaults[i]).upper())
            except:
                raise TypeError("Defaults and/or Components do not contain necessary information.")
    """
    Choose which components to randomly have faults injected. 

    Parameter:
        chance (float): the chance that each component will fault.

    Returns:
        toInject (list[bool]): Indicates whether a fault is to be injected to the component at this index 
    """
    def randomize(self, chance : float, **kwargs):
        if not 0. <= chance <= 1.:
            raise ValueError("Chance must be between 0 and 1.")
        seed = kwargs.get("seed")
        if seed is not None:
            random.seed(seed)
        toInject = [random.random() < chance for _ in self.components]
        return toInject
    