import h5py

# Specify the path to the .h5 file using raw string notation
h5_file_path = r'F:\trading\improvised-code-of-the-pdf-GPT-main\data\sentiment_model.h5'

# Open the .h5 file
try:
    with h5py.File(h5_file_path, 'r') as h5_file:
        print("Contents of the HDF5 file:")
        
        # Loop through each item in the file
        def printname(name):
            print(name)

        # Print the names of all groups and datasets
        h5_file.visit(printname)

        # Iterate over top-level keys and check their types
        for key in h5_file.keys():
            print(f"\nDataset/Group: {key}")
            item = h5_file[key]
            if isinstance(item, h5py.Group):
                print(f"{key} is a group.")
            elif isinstance(item, h5py.Dataset):
                print(f"{key} is a dataset with data: {item[:]}")
            else:
                print(f"{key} is of unknown type.")

except FileNotFoundError:
    print(f"File not found: {h5_file_path}")
except Exception as e:
    print(f"An error occurred: {e}")
