import json

# Path to the JSON file
file_path = "C:/Users/thoma/OneDrive/Documents/EPFL/MA2/Deep Learning for Autonomous Vehicles/DLAV/3DMPPE_ROOTNET_RELEASE/data/Human36M/annotations/Human36M_subject5_data.json"

# Open the JSON file
with open(file_path, 'r') as file:
    # Load the JSON data
    data = json.load(file)

# Open the file in write mode and save the string
with open("output.txt", "w") as file:
    file.write(str(data))
