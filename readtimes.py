import os

folder_path = "data/intp3003/"  # Replace this with the path to your folder
time_values = []

# List all files in the folder
files = os.listdir(folder_path)

# Iterate through each file
for file_name in files:
    # Split the filename by underscore and dot to extract the time value
    parts = file_name.split('_')
    if len(parts) == 2:
        time_value = parts[1].split('.d')[0]  # Remove the extension
        print(time_value,parts[1].split('.d')[1])
        time_values.append(time_value)

print("Time values:", time_values)
