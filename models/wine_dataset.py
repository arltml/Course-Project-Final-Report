import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from models.wine_sample import WineSample

class WineDataset:
    """
    Manage collection of WineSample objects and performs classification
    """

    def __init__(self):
        """
        Initialize an empty dataset
        """
        self.samples = []
        self.df = None
        self.model = None