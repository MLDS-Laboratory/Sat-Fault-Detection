import random
from abc import ABC, abstractmethod
from collections.abc import Collection
from typing import Any


class Fault(ABC):
    """
    Abstract base class for componentwise fault injection
    """
    def __init__(self, components: Collection[Any], **kwargs):
        """
        Initialize the abstract fault. 

        Parameters:
            components (Collection): the components in which to (potentially) inject the fault
            defaults: a dictionary of the default settings for each component
        """
        self.components = components
        self.defaults = kwargs.get("defaults") if kwargs.get("defaults") else None
    
    @abstractmethod
    def inject(self) -> Collection[Any]:
        """
        Inject a predefined fault. 

        Returns:
            Collection: Details of fault injected into each component.
        """
        pass

    @abstractmethod
    def inject_random(self, chance : float) -> Collection[tuple[bool, Any]]:
        """
        Inject random faults into random components.

        Parameter:
            chance (float): the chance that each component will fault.

        Returns:
            Collection:
                bool: Indicates whether a fault is to be injected to the component at this index 
                Any: Details of fault injected, None if no fault injected.  
        """
        pass

    @abstractmethod
    def reset(self) -> bool:
        """
        Reset all components to settings pre-fault injections, if defaults are given. 
        
        Returns:
            bool: True if successful
        """
        pass

    def randomize(self, chance : float, **kwargs):
        """
        Choose which components to randomly have faults injected. 

        Parameter:
            chance (float): the chance that each component will fault.

        Returns:
            toInject (list[bool]): Indicates whether a fault is to be injected to the component at this index 
        """
        if not 0 <= chance <= 1:
            raise ValueError("Chance must be between 0 and 1.")
        seed = kwargs.get("seed")
        if seed is not None:
            random.seed(seed)
        toInject = [random.random() < chance for _ in self.components]
        return toInject



