import pandas as pd

def load_wine_data(filepath:str):
    """
    Load wine dataset from CSV file
    """
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print("Error: File not found.")
        raise
    except Exception as e:
        print(f"Error loading file:{e}")
        raise