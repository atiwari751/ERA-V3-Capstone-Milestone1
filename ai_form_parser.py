import pandas as pd
import requests
import io

# Set pandas display options for anyone who might want to print the dataframe directly
pd.set_option('display.width', 2000)
pd.set_option('display.max_columns', 20)

def _fetch_and_prepare_data(sheet_url: str):
    """
    (Internal function) Fetches and prepares data from the Google Sheet.
    This includes setting correct data types as specified.
    """
    try:
        response = requests.get(sheet_url)
        response.raise_for_status()
        csv_file = io.StringIO(response.content.decode('utf-8'))
        
        # Use the SECOND row (index 1) as the header
        df = pd.read_csv(csv_file, header=1)
        
        # --- Data Type Casting as per requirements ---
        
        # Define the types for each column
        data_types = {
            'Extents X (m)': 'int', 'Extents Y (m)': 'int',
            'Grid Spacing X (m)': 'int', 'Grid Spacing Y (m)': 'int',
            'No. of floors': 'int', 'Steel tonnage (tons/m2)': 'float',
            'Column size (mm)': 'int', 'Structural depth (mm)': 'int',
            'Concrete tonnage (tons/m2)': 'float'
        }
        
        for col, dtype in data_types.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle the boolean column separately
        if 'Trustworthiness' in df.columns:
            # --- THIS IS THE FIX ---
            # First, cast the column to string type to handle mixed data (booleans, NaNs, etc.).
            # Then, perform the string operations. This is much more robust.
            df['Trustworthiness'] = df['Trustworthiness'].astype(str).str.strip().str.upper() == 'TRUE'
        
        # Now, cast to the specific integer/float types
        df = df.astype(data_types, errors='ignore')
        
        # Drop rows where a critical value might be missing
        df.dropna(subset=['Extents X (m)', 'No. of floors'], inplace=True)
        
        print("✅ Data successfully downloaded and prepared.")
        return df

    except requests.exceptions.RequestException as e:
        print(f"\nError: Could not fetch the Google Sheet. Details: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while processing the data: {e}")
        return None

class AiFormFinder:
    """
    A class to find AiForm outputs based on inputs, using data from a Google Sheet.
    The data is fetched once upon initialization for efficient lookups.
    """
    def __init__(self, sheet_url: str):
        """
        Initializes the finder by downloading and preparing the data.
        
        Args:
            sheet_url (str): The public export URL of the Google Sheet.
        """
        self.data = _fetch_and_prepare_data(sheet_url)
        if self.data is not None:
            self.is_ready = True
            self.output_columns = [
                'Steel tonnage (tons/m2)', 'Column size (mm)', 
                'Structural depth (mm)', 'Concrete tonnage (tons/m2)', 'Trustworthiness'
            ]
        else:
            self.is_ready = False
            # The error message is now printed inside the fetch function

    def get_output(self, input_data: dict) -> dict | None:
        """
        Finds the output data for a given dictionary of input parameters.

        Args:
            input_data (dict): A dictionary with the 5 input parameters as keys.

        Returns:
            dict: A dictionary containing the 5 output values if a match is found.
            None: If no match is found or if the finder is not ready.
        """
        if not self.is_ready:
            print("❌ Cannot perform lookup: data was not loaded successfully.")
            return None

        try:
            # Build the query condition from the input dictionary
            query_condition = (self.data['Extents X (m)'] == input_data['Extents X (m)']) & \
                              (self.data['Extents Y (m)'] == input_data['Extents Y (m)']) & \
                              (self.data['Grid Spacing X (m)'] == input_data['Grid Spacing X (m)']) & \
                              (self.data['Grid Spacing Y (m)'] == input_data['Grid Spacing Y (m)']) & \
                              (self.data['No. of floors'] == input_data['No. of floors'])
            
            result_df = self.data[query_condition]

            if not result_df.empty:
                result_series = result_df.iloc[0]
                output_dict = result_series[self.output_columns].to_dict()
                return output_dict
            else:
                return None
        except KeyError as e:
            print(f"Error: Input dictionary is missing a required key: {e}")
            return None
        except Exception as e:
            print(f"An error occurred during the lookup: {e}")
            return None

# --- Example of how to use the AiFormFinder class ---
if __name__ == "__main__":
    
    sheet_url = 'https://docs.google.com/spreadsheets/d/1ukC2UCcFznkPZMHlbQu48dmd61hwme4DmJSGsjtLyWE/export?format=csv&gid=0'
    
    # 1. Create an instance of the finder.
    finder = AiFormFinder(sheet_url)

    # 2. Check if the finder is ready before using it.
    if finder.is_ready:
        print("\n--- Example 1: Finding a matching record ---")
        # Define the inputs in a dictionary
        input_1 = {
            'Extents X (m)': 23,
            'Extents Y (m)': 23,
            'Grid Spacing X (m)': 6,
            'Grid Spacing Y (m)': 6,
            'No. of floors': 3
        }
        
        # 3. Call the get_output method
        output_1 = finder.get_output(input_1)
        
        if output_1:
            print("✅ Match Found! Output dictionary:")
            print(output_1)
            # You can also access individual items
            print(f"Trustworthiness from result: {output_1['Trustworthiness']} (Type: {type(output_1['Trustworthiness'])})")
        else:
            print("❌ No match found for input 1.")

        print("\n--- Example 2: A record that does not exist ---")
        input_2 = {
            'Extents X (m)': 100, 'Extents Y (m)': 100,
            'Grid Spacing X (m)': 5, 'Grid Spacing Y (m)': 5,
            'No. of floors': 1
        }
        output_2 = finder.get_output(input_2)
        
        if output_2:
            print("✅ Match Found! Output dictionary:")
            print(output_2)
        else:
            print("❌ No match found for input 2, as expected.")
