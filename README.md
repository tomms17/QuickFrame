# QuickFrame: A Data Analysis and Preprocessing Toolkit

## Overview
QuickFrame is a Python class that extends the functionality of the pandas.DataFrame class with additional methods for data analysis and preprocessing. This class provides convenient tools for generating correlation matrices, counting NaN values, plotting column distributions, and performing data preprocessing tasks such as handling missing values, dropping highly correlated features, encoding categorical variables, and scaling numerical features.

## Installation
To use QuickFrame, make sure you have the required dependencies installed. You can install them using the following:

## Example: Initialize QuickFrame from data
data = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}  
quick_frame = QuickFrame(data)

## Example: Initialize QuickFrame from a CSV file
csv_file_path = 'data.csv'  
quick_frame_csv = QuickFrame(csv_file=csv_file_path)

## Example: Generate a normalized correlation matrix heatmap
quick_frame.norm_corr_matrix(scale_data='MinMaxScaler')

## Example: Count NaN values per column
quick_frame.nan_per_column()

## Example: Plot distributions of specified columns
quick_frame.plot_distributions(x_axis='A', exclude='B')

## Example: Preprocess data
preprocessed_data = quick_frame.preprocess_data(  
    set_index_as='A',  
    corr_threshold=0.5,  
    encode=True,  
    impute_modes={'B': 'mean', 'C': 'mode'},  
    scale_data='StandardScaler'  
)

## Class Methods
__init__(self, data: Union[dict, List[str], np.ndarray] = None, csv_file=None, **kwargs)
Initialize the QuickFrame by creating a DataFrame either from data or a CSV file.

norm_corr_matrix(self, scale_data: str = None)
Generate a heatmap of the normalized correlation matrix for the DataFrame.

nan_per_column(self)
Count the number of NaN values in each column of the DataFrame.

plot_distributions(self, x_axis: str = 'index', exclude: Union[str, List[str]] = None)
Plot distributions of specified columns for each country in the DataFrame.

preprocess_data(self, set_index_as: Union[str, List[str]] = None, corr_threshold: float = None, encode: bool = False, impute_modes: dict = None, scale_data: str = None)
Preprocess the DataFrame by handling missing values, dropping highly correlated features, and optionally encoding categorical variables and scaling numerical features.

## Parameters and Returns
Refer to the method docstrings for detailed information on parameters and return values.

## Requirements
Python 3.6+
pandas
numpy
matplotlib
seaborn
scikit-learn

## License
This code is provided under the MIT License. Feel free to use, modify, or distribute it according to your needs.
