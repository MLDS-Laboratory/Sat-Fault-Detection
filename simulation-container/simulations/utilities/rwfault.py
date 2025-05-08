
from collections.abc import Collection
from collections import OrderedDict
from typing import Any
import random
from Basilisk.architecture import messaging, sysModel
import numpy as np
"""
This is a module for introducing faults into Reaction Wheels. It is a Basilisk module,
so it runs at every timestep provided it is added to a simulation task. 

The only implemented methodology for faults thus far is a reduction in max torque output 
of the reaction wheels. The introduction of friction will eventually be added. 
"""
class RWFault(sysModel.SysModel):
    """
    Requires, in order:
      - A list of the form (type, RW), where type is the model of the reaction wheel and RW is the reaction wheel itself
      - the rwFactory (simIncluderwFactory()) used to generate those wheels
      - the rwEffector (reactionWheelStateEffector) used to make the wheels actually do something
    
    Optional:
      - the model used for the RWs (jitter, balanced, etc). not not implemented yet
      - a seed for the randomized fault injection
      - a fault injection chance per RW for the randomized fault injection
      - defaults - the original settings of the RWs
    """
    def __init__(self, components: Collection[Any], rwFactory, rwEffector, **kwargs):
        super().__init__()
        self.components = components
        self.rwFactory = rwFactory
        self.rwEffector = rwEffector
        self.rwModel = kwargs.get("model")
        self.seed = kwargs.get("seed")
        self.chance = kwargs.get("chance")
        self.defaults = kwargs.get("defaults")
        self.friction = kwargs.get("friction")
        self.types = kwargs.get("types")
        if not self.types:
            self.types = ["torque"]
        #if the defaults are not passed in, the defaults are pulled from the starting states
        if not self.defaults:
            self.defaults = []
            for i in range(len(components)):
                spin = [sublist[0] for sublist in self.components[i][1].gsHat_B]
                rwType = self.components[i][0]
                u_max = self.components[i][1].u_max
                cvis = self.components[i][1].cViscous
                self.defaults.append({"axis":spin, "u_max":u_max, "rwType":rwType, "cViscous":cvis})
        #loggable attribute for the actual traits of the RWs
        self.state = [(i["rwType"], i["axis"], i["u_max"], i["cViscous"]) for i in self.defaults]
        #loggable attribute for the timesteps of fault injections
        self.fault = [False for _ in self.components]
        self.count = [0] * len(self.components)

    """
    Update method. Runs every timestep. 

    Just calls the inject_random method, and updates the loggable attributes with the results. 
    """
    def UpdateState(self, CurrentSimNanos):
        state = self.inject_random(self.chance)
        self.state = state[:, 1]
        self.fault = state[:, 0]
        self.count = [self.count[i] + 1 if self.fault[i] else self.count[i] for i in range(len(self.components))]
    

    """
    The method that actually changes out the RWs. 

    Only called from inject() - takes in new RW settings, and uses them to replace
    the batch of RWs that existed previously with new ones with those settings.

    This is the only way I could find to introduce faults into RWs on the Python side. 
    """
    def replaceRWs(self, newRWs):
        #empty existing RW carriers

        self.rwFactory.rwList = OrderedDict([])
        self.rwEffector.ReactionWheelData = self.rwEffector.ReactionWheelData[:0]
        newComps = []
        #create new RWs
        for (rwType, spin, torque, cvis) in newRWs:
            rwModel = self.rwModel if self.rwModel else messaging.BalancedWheels
            RW_new = self.rwFactory.create(rwType, spin, maxMomentum=100., RWmodel=rwModel, u_max=torque, 
                                            cViscous=cvis)
            self.rwEffector.addReactionWheel(RW_new)
            newComps.append((rwType, RW_new))
        self.components = newComps
        
    """
    Resets all RWs to default settings. 
    """
    def reset(self):
        if not self.defaults:
            raise ValueError("Defaults have not been set.")
        resetRWs = []
        try:
            for i in range(len(self.components)):
                spin = self.defaults[i]["axis"]
                u_max = self.defaults[i]["u_max"]
                rwType = self.defaults[i]["type"]
                cvis = self.defaults[i]["cViscous"]
                resetRWs.append((rwType, spin, u_max, cvis))
        except:
            raise TypeError("Defaults and/or Components do not contain necessary information.")
        self.replaceRWs(resetRWs)
        return len(self.rwFactory.rwList) > 0 and len(self.rwEffector.ReactionWheelData) > 0

    """
    Meant for non-random injections where the new desired settings are known. Not used in practice, 
    yet, except as a bridge between inject_random and replaceRWs. 
    """
    def inject(self, newSettings) -> Collection[Any]:
        self.replaceRWs(newSettings)
        return newSettings
    """
    Does the bulk of the injections in practice currently, randomized. 

    Each RW has the same chance of being faulted. If it isn't faulted, 
    the RW still will be replaced, but with the same settings. 
    If it is faulted, the RW's max torque output will be reduced by some 
    random factor. This may become multiplicative. 

    This method returns a list of tuples of the form:
    (bool, (rwType, spin axis, torque)) 
    where bool is whether a fault was injected, rwType is the rwModel, and torque is the max torque output. 
    """
    def inject_random(self, chance : float) -> Collection[tuple[bool, Any]]:
        #if chance isn't passed in

        if chance is None:
            chance = 0.001
        toInject = self.randomize(chance, seed=self.seed)
        if self.seed:
            random.seed(self.seed)
        newSettings = []
        for i in range(len(toInject)):
            #spin and rwType stay the same
            spin = [sublist[0] for sublist in self.components[i][1].gsHat_B]
            rwType = self.components[i][0]
            #u_max and/or cvis is changed if there's a fault by a random factor
            u_max = self.components[i][1].u_max
            cvis = self.components[i][1].cViscous
            if toInject[i]:
                rand = random.random() * 10
                if "torque" in self.types:
                    u_max = u_max / rand if rand >= 1 else u_max * rand
                if "friction" in self.types:
                    cvis = cvis / rand if rand < 1 else cvis * rand
            newSettings.append((rwType, spin, u_max, cvis))
        details = list(zip(toInject, self.inject(newSettings)))
        return np.array(details, dtype=object)
    
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
        