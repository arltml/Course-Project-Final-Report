class WineSample:
    """
    Represents single wine sample along with its chemcial properties
    """
    def __init__(self, features: dict, label: int = None):
        """
        Initialize a wine sample object
        """
        self.features = features
        self.label = label
        
    def __str__(self):
        "Readable string representation of the wine sample."
        return f"WineSample(features ={self.features}, label={self.label})"
    
    def __eq__(self, other):
        """
        Comparison of two WineSample objects based on their features.
        """
        if not isinstance(other, WineSample):
            return False
        return self.features == other.features
    
    def get_feature_vector(self):
        """
        features are returned as an immutable tuple
        """
        return tuple(self.features.values())