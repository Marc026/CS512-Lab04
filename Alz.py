import pandas as pd
import numpy as np

class AlzheimerDataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.load_data()
        print("The data is now inserted into a 2-dimensional array")
    
    def load_data(self):
        df = pd.read_csv(self.file_path, header=None, na_values=['', ' ', 'NaN', 'None'])
        df = df.apply(pd.to_numeric, errors='coerce')
        return df.to_numpy()

    def remove_high_junk_columns(self, junk_threshold=0.7):
        # Using numpy vectorized operations to handle junk columns
        junk_mask = np.isnan(self.data).mean(axis=0) >= junk_threshold
        deleted_columns = np.where(junk_mask)[0]
        self.data = self.data[:, ~junk_mask]
        print("\nColumns with at least 70% junk removed:")
        print(f"{', '.join(['X' + str(col) for col in deleted_columns])} deleted.\n")

    def remove_high_zero_columns(self, zero_threshold=0.9):
        print("\nRemoving columns with at least 90% zeros:")
        high_zero_mask = (self.data == 0).mean(axis=0) >= zero_threshold
        deleted_columns = np.where(high_zero_mask)[0]
        self.data = self.data[:, ~high_zero_mask]
        print(f"{', '.join(['X' + str(col) for col in deleted_columns])} deleted.")

    def remove_rows_with_junk(self, junk_threshold=0.7):
        print("\nRemoving rows with at least 70% junk:")
        junk_mask = np.isnan(self.data).mean(axis=1) >= junk_threshold
        deleted_rows = np.where(junk_mask)[0]
        self.data = self.data[~junk_mask]
        print(f"{', '.join(map(str, deleted_rows))} deleted.")

    def remove_rows_without_ic50(self, ic50_index):
        print("\nRemoving rows without IC50 (Y value):")
        ic50_mask = np.isnan(self.data[:, ic50_index])
        deleted_rows = np.where(ic50_mask)[0]
        self.data = self.data[~ic50_mask]
        print(f"{', '.join(map(str, deleted_rows))} deleted.")

    def replace_junk_with_median(self):
        print("\nFor any cell that is still junk, replace the value with the median of the column:")
    
    # Calculate the median for each column, ignoring NaN values
        medians = np.nanmedian(self.data, axis=0)

    # Create a mask for NaN values in the data
        junk_mask = np.isnan(self.data)

    # Replace NaN values with the corresponding column median
    # Use broadcasting to align the medians with the shape of the junk mask
        self.data[junk_mask] = medians[np.where(junk_mask)[1]]  # Replace NaNs with corresponding column medians

    # Get indices of replaced cells for printing
        indices = np.argwhere(junk_mask)

    # Print out the replacements
        replacements = [f"Cell ({idx[0]}, {idx[1]}) replaced with median value {medians[idx[1]]}" for idx in indices]
        print('\n'.join(replacements))

    def remove_remaining_nans(self):
    # Using numpy's built-in nan-handling to remove rows with any NaN
        nan_mask = np.isnan(self.data).any(axis=1)
        deleted_rows = np.where(nan_mask)[0]
        self.data = self.data[~nan_mask]

    # Relying on Python's default behavior to print only if deleted_rows has elements
        deleted_rows_list = ', '.join(map(str, deleted_rows))
        print(f"{deleted_rows_list} deleted due to remaining NaNs") if deleted_rows_list else None

    def rescale_data(self):
        min_values = np.nanmin(self.data, axis=0)
        max_values = np.nanmax(self.data, axis=0)
        ranges = max_values - min_values
        ranges[ranges == 0] = 1  # Avoid division by zero
        self.data = (self.data - min_values) / ranges
        print("\nData has been rescaled.")

    def normalize_data(self):
        mean = np.nanmean(self.data, axis=0)
        std = np.nanstd(self.data, axis=0)
        std[std == 0] = 1  # Prevent division by zero
        self.data = (self.data - mean) / std
        print("\nData has been normalized.")

    def save_to_csv(self, original_data, cleaned_data, rescaled_data, normalized_data):
    # Create a Pandas Excel writer using openpyxl as the engine
        with pd.ExcelWriter('Alzheimer_Data_Combined.xlsx', engine='openpyxl') as writer:
        # Save original data to the first tab
            pd.DataFrame(original_data).to_excel(writer, sheet_name='Original Data', index=False, header=False)

        # Save cleaned data to the second tab
            pd.DataFrame(cleaned_data).to_excel(writer, sheet_name='Cleaned Data', index=False, header=False)

        # Save rescaled data to the third tab
            pd.DataFrame(rescaled_data).to_excel(writer, sheet_name='Rescaled Data', index=False, header=False)

        # Save normalized data to the fourth tab
            pd.DataFrame(normalized_data).to_excel(writer, sheet_name='Normalized Data', index=False, header=False)

if __name__ == "__main__":
    processor = AlzheimerDataProcessor('Alzheimer2.csv')
    
    original_data = processor.data.copy()

    processor.remove_high_junk_columns()
    processor.remove_high_zero_columns()
    processor.remove_rows_with_junk()
    processor.remove_rows_without_ic50(ic50_index=0)
    processor.replace_junk_with_median()

    processor.remove_remaining_nans()
    
    cleaned_data = processor.data.copy()
    
    processor.rescale_data()
    rescaled_data = processor.data.copy()
    
    processor.normalize_data()
    normalized_data = processor.data.copy()
    
    processor.save_to_csv(original_data, cleaned_data, rescaled_data, normalized_data)
