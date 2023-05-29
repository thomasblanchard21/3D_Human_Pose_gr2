import json

# Path to the JSON file
file_path = "C:/Users/thoma/OneDrive/Documents/EPFL/MA2/Deep Learning for Autonomous Vehicles/DLAV/3DMPPE_POSENET_RELEASE/data/Human36M/annotations/Human36M_subject1_joint_3d.json"

# Open the JSON file
with open(file_path, 'r') as file:
    # Load the JSON data
    data = json.load(file)

# Open the file in write mode and save the string
with open("output.txt", "w") as file:
    file.write(str(data))