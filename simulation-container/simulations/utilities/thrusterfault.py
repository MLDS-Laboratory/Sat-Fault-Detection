from collections.abc import Collection
from collections import OrderedDict
from typing import Any
import random
from Basilisk.architecture import messaging, sysModel
import numpy as np

class ThrusterFault(sysModel.SysModel):
    #simulation, simulation timestep, simIncludeThruster.thrusterFactory(), thrusterDynamicEffector.ThrusterDynamicEffector(), location relative to body origin, thrust pointing vector in body frame
    def __init__(self, sim, timestep, thFactory, thSet, locations, directions, **kwargs):
        super().__init__()
        self.sim = sim
        self.components = thFactory.thrusterList
        self.timestep = timestep
        self.thFactory = thFactory
        self.thSet = thSet
        self.seed = kwargs.get("seed")
        self.chance = kwargs.get("chance")
        self.type = kwargs.get("type")
        if not self.type:
            self.type = 'Blank_Thruster'
        self.locations = locations
        self.directions = directions
        self.thrTimeData = messaging.THRArrayOnTimeCmdMsgPayload()
        self.thrTimeData.OnTimeRequest = np.zeros(len(self.components))
        self.thrTimeMsg = messaging.THRArrayOnTimeCmdMsg().write(self.thrTimeData)
        self.thrusterSet.cmdsInMsg.subscribeTo(self.thrTimeMsg)
        self.state = [(False, (l, d, 0, 0)) for (l, d) in zip(locations, directions)]
    
    def UpdateState(self, CurrentSimNanos):
        self.state = self.inject_random(self.chance, CurrentSimNanos)

    def replaceThrusters(self, newSettings, currentTime):
        self.thFactory.thrusterList = OrderedDict([])
        self.thrusterSet.thrusterData = self.thrusterSet.thrusterData[:0]
        thrustTime = []
        for (pos, direct, force, time) in newSettings:
                temp = self.thFactory.create('Blank_Thruster', pos, direct, MaxThrust=force)
                self.thrusterSet.addThruster(temp)
                thrustTime.append(time)
        self.thrTimeData.OnTimeRequest = thrustTime
        self.thrTimeMsg.write(self.thrTimeData, time=currentTime)
        self.components = self.thFactory.thrusterList
        

    def inject(self, newSettings, currentTime):
        self.replaceThrusters(newSettings, currentTime)
        return newSettings

    def inject_random(self, chance, currentTime):
        if chance is None:
            chance = 0.001
        toInject = self.randomize(chance, seed=self.seed)
        forces = []
        thrustTime = []
        for i in range(len(self.components)):
            randf = random.random() * 500.
            randt = random.random() * self.timestep + self.timestep 
            #^^last between 1 and 2 timesteps, so it shows up on the plot
            if toInject[i]:
                forces.append(randf)
                thrustTime.append(randt)
            else:
                forces.append(500.)
                thrustTime.append(0.)
        newSettings = zip(self.locations, self.directions, forces, thrustTime)
        self.inject(newSettings, currentTime)
        return np.array(zip(toInject, newSettings), dtype=object)
    
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
