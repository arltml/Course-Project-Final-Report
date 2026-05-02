import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from wine_sample import WineSample

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

    def load_data(self, df: pd.DataFrame):
        """
        Loads dataset from Pandas DataFrame and creates WineSample Objects
        """
        self.df = df
        self.samples = []
        #Loop for creating dataset with rows
        for _, row in df.iterrows():
            features = row.drop("target").to_dict()
            label = row["target"]
            sample = WineSample(features, label)
            self.samples.append(sample)
    def __len__(self):
        """
        return the number of samples within the dataset as an integer
        """
        return len(self.samples)
    
    def train_model(self):
        """
        Uses and trains KNN classifier using the provided database.
        """
        if self.df is None:
            raise ValueError("Dataset not loaded.")

        X = self.df.drop("target", axis=1)
        y = self.df["target"]

        self.model = KNeighborsClassifier(n_neighbors=3)
        self.model.fit(X, y)
    
    def predict(self, sample:WineSample):
        """
        Predict the class of a wine sample
        """
        if self.model is None:
            raise ValueError("Model not trained.")
        
        feature_vector = [list(sample.features.values())]
        return self.model.predict(feature_vector)[0]
    
    def get_summary_stats(self):
        """
        Compute summary statistics for dataset.
        """
        if self.df is None:
            raise ValueError("Dataset not loaded.")
        return{
            col: self.df[col].mean()
            for col in self.df.columns if col != "target"
        }