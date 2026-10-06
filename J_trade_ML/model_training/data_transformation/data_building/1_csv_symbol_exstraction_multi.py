import os

# Define the directory path
directory_path = r"C:\Users\crgon\OneDrive\Desktop\trading\data\main_data"

# List to store the unique first parts of filenames
unique_names = set()

# Iterate over each file in the directory
for filename in os.listdir(directory_path):
    if os.path.isfile(os.path.join(directory_path, filename)):
        # Extract the first part of the filename before the first underscore
        first_part = filename.split('_')[0]
        # Add the first part to the set to avoid duplicates
        unique_names.add(first_part)

# Join the unique names with a comma and no spaces, and print them
print(','.join(unique_names))
