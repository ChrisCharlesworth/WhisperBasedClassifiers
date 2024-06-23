import os
import json

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
grandparent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
file_path = os.path.join(parent_dir, "MSDM/train_data.list")



# Path to the .list file
# Initialize the dictionary to hold lists of entries by label
label_dict = {0: [], 1: [], 2: [], 3: []}

# Read the data from the .list file line by line
with open(file_path, 'r') as file:
    for line in file:
        entry = json.loads(line.strip())
        label = entry['label']
        if label in label_dict:
            label_dict[label].append(entry['wav'])

# Limit each label's list to the first 4956 entries
limited_data = []
for label in label_dict:
    limited_data.extend(label_dict[label][:4956])

# Extract only the names without the file path and extension
limited_data = [os.path.splitext(os.path.basename(wav))[0] + ".json" for wav in limited_data]

# Print the total number of entries to verify
print(f"Total number of entries: {len(limited_data)}")



directory = os.path.join(parent_dir, "MSDM_Embeddings_2")
# Initialize an empty dictionary to hold combined data
# combined_data = {}

combined_data = []

# Loop through each file in the list
for file_name in limited_data:
    file_path = os.path.join(directory, file_name)
    with open(file_path, 'r') as file:
        # Load JSON data from the file
        data = json.load(file)
        # Append data to combined_data list
        combined_data.append(data)

# Path for the output combined JSON file
output_file_path = 'MSDM_Subset.json'

# Write combined data to the output file
with open(output_file_path, 'w') as output_file:
    json.dump(combined_data, output_file, indent=4)

print("Combined JSON file created successfully at:", output_file_path)