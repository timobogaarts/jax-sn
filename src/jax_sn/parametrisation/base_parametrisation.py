import equinox as eqx
from abc import ABC, abstractmethod
import lineax as lx
from typing import Union, Callable, Dict

class ParametrizedModule(ABC):

    @staticmethod
    @abstractmethod
    def theta_base(self, module):
        ...

    @staticmethod
    @abstractmethod
    def reconstruct_operator(self, module, theta):
        ...
    
    