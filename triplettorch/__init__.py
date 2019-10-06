"""The Init File for triplettorch module

The file imports all classes from the triplettorch module to expose them to
the user for easier API usage.
It provides a TripletDataset class as well as a TripletMiner class with two
common types of Triplet Miner: AllTripletMiner and HardNegativeTripletMiner.
"""
from triplettorch.loss import HardNegativeTripletMiner
from triplettorch.loss import AllTripletMiner
from triplettorch.data import TripletDataset
from triplettorch.loss import TripletMiner

__name__ = "TripletTorch"
__version__ = "0.1.2"
__author__ = "yliess"
__url__ = "https://github.com/TowardHumanizedInteraction/TripletTorch"
__email__ = "hatiyliess86@gmail.com"