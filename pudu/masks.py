import numpy as np
import random

class Percentage:
    """
    Initializes the Percentage class with a specific percentage.

    :param percentage: float
    The percentage at which the section will be evaluated. 
    """
    def __init__(self, percentage=0.2):
        self.percentage = percentage
    def apply(self, section, total):
        if section <= int(self.percentage*total):
            return 1
        else:
            return 0

class Quantity:
    """
    Initializes the Quantity class with a specific quantity.

    :param qty: int
    The quantity at which the section will be evaluated.
    """
    def __init__(self, qty=1):
        self.qty = qty
    def apply(self, section, total):
        if section <= self.qty:
            return 1
        else:
            return 0

class EveryOther:
    """
    Initializes the EveryOther class with a specific state.

    :param is_eo: int
    The state of the class.
    """
    def __init__(self, is_eo=1):
        self.is_eo = is_eo
    def apply(self, section, total):
        self.is_eo = (self.is_eo + 1) % 2
        return self.is_eo

class Pairs:
    """
    Returns 1 if the given section is even, None otherwise.
    """
    def apply(self, section, total):
        if section%2 == 0:
            return 1
        else:
            return 0

class Odds:
    """
    Returns 1 if the given section is odd, None otherwise.
    """   
    def apply(self, section, total):
        if section%2 != 0: 
            return 1
        else:
            return 0

class RandomMask():
    """
    Returns a random integer between 0 and 1.
    """
    def apply(self, section, total):
        return random.randint(0,1)

class Custom:
    """
    Initializes the Custom class with a specific vector.

    :param vector: list
    The vector to be used in the apply method.

    :return: None
    """
    def __init__(self, vector):
        self.vector = vector
    def apply(self, section, total):
        return self.vector[section]

class All:
    """
    Always returns 1, regardless of the input parameters.
    """
    def apply(self, section, total):
        return 1
