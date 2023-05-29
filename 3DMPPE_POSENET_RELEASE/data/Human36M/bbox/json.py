import json

# Path to the JSON file
file_path = "C:/Users/thoma/OneDrive/Documents/EPFL/MA2/Deep Learning for Autonomous Vehicles/DLAV/3DMPPE_ROOTNET_RELEASE/data/Human36M/bbox/bbox_human36m_output.json"

# Open the JSON file
with open(file_path, 'r') as file:
    # Load the JSON data
    data = json.load(file)

# Print the contents of the JSON file
print(data)
