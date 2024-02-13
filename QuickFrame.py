from typing import Union, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


class QuickFrame(pd.DataFrame):
    """
    A class extending pd.DataFrame with additional functionality for data analysis and preprocessing.

    Parameters:
    - data (dict, list, or ndarray, optional): Data to initialize the DataFrame. Defaults to None.
    - csv_file (str, optional): Path to the CSV file to read data from. Defaults to None.
    - **kwargs: Additional arguments to pass to the pd.DataFrame constructor.

    Methods:
    - norm_corr_matrix(): Generate a heatmap of the normalized correlation matrix.
    - nan_per_column(): Count the number of NaN values in each column.
    - plot_distributions(x_axis='index'): Plot distributions of specified columns.
    - preprocess_data(set_index_as=None, corr_threshold=None, encode=True,
                      impute_modes=None, scale_data=None): Preprocess the DataFrame.
    """

    def __init__(self, data=None, csv_file=None, **kwargs):
        """
        Initialize the QuickFrame by creating a DataFrame either from data or a CSV file.

        Parameters:
        - data (dict, list, or ndarray, optional): Data to initialize the DataFrame. Defaults to None.
        - csv_file (str, optional): Path to the CSV file to read data from. Defaults to None.
        - **kwargs: Additional arguments to pass to the pd.DataFrame constructor.
        """
        if csv_file is not None:
            data = pd.read_csv(csv_file, **kwargs)
        super().__init__(data)

    def norm_corr_matrix(self):
        """
        Generate a heatmap of the normalized correlation matrix for the DataFrame.

        Raises:
        - ValueError: If the DataFrame is empty.

        Returns:
        - None: Displays a heatmap of the normalized correlation matrix.
        """
        # add option to normalize (same as in preprocess), not inplace!!!
        correlation_matrix = self.corr(numeric_only=True)
        normalized_corr_matrix = (correlation_matrix - correlation_matrix.min().min()) / (
                correlation_matrix.max().max() - correlation_matrix.min().min()) * 2 - 1

        plt.figure(figsize=(10, 8))
        sns.heatmap(normalized_corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', linewidths=0.5)
        plt.title("Normalized Correlation Matrix (-1 to 1)")
        plt.show()

    def nan_per_column(self):
        """
        Count the number of NaN values in each column of the DataFrame.

        Returns:
        - None: Prints the NaN count per column.
        """
        nan_count_per_column = self.isna().sum()
        print("NaN count per column:")
        print(nan_count_per_column)

    def plot_distributions(self, x_axis: str = 'index', exclude: Union[str, List[str]] = None):
        """
        Plot distributions of specified columns for each country in the DataFrame.

        Parameters:
        - x_axis (str, optional): Specify the x-axis for scatter plots. Default is 'index'.
        - exclude (str or list, optional): Columns to exclude from the plots. Default is None.
            If a single column name is provided, it will be converted to a list internally.

        Raises:
        - KeyError: If the specified x_axis column does not exist in the DataFrame.
        - TypeError: If the exclude parameter is not a valid type (str or list).

        Returns:
        - None: Displays scatter plots for specified columns.
        """
        if x_axis not in self.columns:
            raise KeyError(f"The specified x_axis column '{x_axis}' does not exist in the DataFrame.")

        if exclude is None:
            exclude = []
        elif not isinstance(exclude, (str, list)):
            raise TypeError(f"Invalid type for 'exclude'. Expected str or list, got {type(exclude).__name__}.")
        if isinstance(exclude, str):
            exclude = [exclude]
        columns_to_plot = list(set(self.columns) - {x_axis} - set(exclude))

        num_rows = (len(columns_to_plot) + 2) // 3
        fig, axes = plt.subplots(nrows=num_rows, ncols=3, figsize=(12, 4 * num_rows))
        axes = axes.flatten()

        i = 0
        if x_axis == 'index':
            for column in columns_to_plot:
                sns.scatterplot(x=self.index, y=column, data=self, ax=axes[i])
                axes[i].set_title(f'{column}')
                axes[i].set_xlabel('Index')
                axes[i].set_ylabel(column)
                i += 1

            for j in range(i, len(axes)):
                fig.delaxes(axes[j])
        else:
            for column in columns_to_plot:
                sns.scatterplot(x=self[x_axis], y=column, data=self, ax=axes[i])
                axes[i].set_title(f'{column}')
                axes[i].set_xlabel(f'{x_axis}')
                axes[i].set_ylabel(column)
                i += 1  # Increment i inside the loop

            for j in range(i, len(axes)):
                fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    def preprocess_data(self, set_index_as: Union[str, List[str]] = None,
                        corr_threshold: float = None, encode: bool = False,
                        impute_modes: dict = None, scale_data: str = None):
        """
        Preprocess the DataFrame by handling missing values, dropping highly correlated features,
        and optionally encoding categorical variables and scaling numerical features.

        Parameters:
        - set_index_as (str, optional): Column name to set as the index.
        - corr_threshold (float, optional): Threshold for dropping highly correlated features.
        - encode (bool, optional): Whether to encode categorical variables. Default is True.
        - impute_modes (dict, optional): Dictionary specifying imputation mode for each column.
                                        Valid values: 'mean', 'mode', 'all_0', 'all_1'.
                                        Function does not impute values if None is passed.
        - scale_data (str, optional): Scaling method for numerical features.
                                     Options: 'MinMaxScaler', 'RobustScaler', 'StandardScaler', or None.
                                     Default is None.

        Raises:
        - ValueError: If the DataFrame is empty, if 'impute_modes' is not provided or not a dictionary,
                      if 'corr_threshold' is not a float, if 'scale_data' is not one of the specified options,
                      or if any value in 'impute_modes' is not one of ['mean', 'mode', 'all_0', 'all_1'].

        Returns:
        - QuickFrame: Preprocessed QuickFrame.
        """
        # add option to use multi level indexing
        if not isinstance(encode, bool):
            raise ValueError("'encode' must be a boolean value")

        if set_index_as is not None and not isinstance(set_index_as, str):
            raise ValueError("'set_index_as' must be a string or None")

        if impute_modes is not None:
            if not isinstance(impute_modes, dict):
                raise ValueError("'impute_modes' must be a dictionary or None")

            valid_impute_methods = ['mean', 'mode', 'all_0', 'all_1', 'median']
            invalid_methods = [method for method in impute_modes.values() if method not in valid_impute_methods]
            if invalid_methods:
                raise ValueError(f"Invalid impute method(s): {', '.join(invalid_methods)}")
        # make it eat int as well !!!
        if corr_threshold is not None and not isinstance(corr_threshold, float):
            raise ValueError("'corr_threshold' must be a float value or None")

        valid_scalers = {'MinMaxScaler': MinMaxScaler,
                         'RobustScaler': RobustScaler,
                         'StandardScaler': StandardScaler}
        scaler_class = valid_scalers.get(scale_data, None)

        def internal_impute_column(series, method):
            impute_methods = {
                'mean': series.mean(),
                'mode': series.mode().iloc[0] if not series.mode().empty else np.nan,
                'all_0': 0,
                'all_1': 1,
                'median': series.median(),
            }

            if method not in impute_methods:
                raise ValueError(
                    f"Invalid imputation method: {method}. Possible values are {list(impute_methods.keys())}.")

            return series.fillna(impute_methods.get(method, series))

        preprocessed_df = QuickFrame(self.copy(deep=True))

        if impute_modes is not None:
            for column, method in impute_modes.items():
                preprocessed_df[column] = internal_impute_column(preprocessed_df[column], method)
        # check reindexing !!!
        if corr_threshold is not None:
            correlation_matrix = preprocessed_df.corr(numeric_only=True).abs()
            upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
            columns_to_drop = [column for column in upper_triangle.columns if
                               any(upper_triangle[column] > corr_threshold)]
            preprocessed_df.drop(columns=columns_to_drop, inplace=True)

        if set_index_as is not None:
            if set_index_as in self.columns:
                preprocessed_df.set_index(set_index_as, inplace=True)
            else:
                raise KeyError("Inputted column does not exist in DataFrame")
        # test for - handling non-numerical values !!!
        if scaler_class is not None:
            numerical_columns = preprocessed_df.select_dtypes(include=['int', 'float']).columns
            preprocessed_df[numerical_columns] = scaler_class().fit_transform(preprocessed_df[numerical_columns])

        if encode:
            columns_to_encode = preprocessed_df.select_dtypes(include=['object', 'category']).columns
            preprocessed_df = pd.get_dummies(preprocessed_df, columns=columns_to_encode)

        boolean_columns = preprocessed_df.select_dtypes(include='bool').columns
        preprocessed_df[boolean_columns] = preprocessed_df[boolean_columns].astype(int)

        return QuickFrame(preprocessed_df)
