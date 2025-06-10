import pandas as pd

def get_top_scenes(num_scenes = 10):
    """
    This function reads a CSV file containing wetland development data and returns the top scenes
    with the highest wetland development in hectares.
    
    Returns:
        pd.DataFrame: A DataFrame containing the top scenes and their corresponding wetland development values.
    """
    # reading tabular data
    cols = ['scene', 'LOG_WETLAND_DEV_1996_2016_HECTARES']
    df = pd.read_csv("/Users/sayedmorteza/Library/CloudStorage/Box-Box/Caltech Research/Scripts/" \
        "WetlandCausalML/tabular_data_florida.csv", usecols=cols)

    # finding the top scenes with the most wetland development
    top_scenes = df.sort_values(by='LOG_WETLAND_DEV_1996_2016_HECTARES', ascending=False).head(num_scenes).copy()

    return top_scenes
