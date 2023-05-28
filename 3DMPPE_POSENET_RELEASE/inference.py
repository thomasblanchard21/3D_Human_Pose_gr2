import torch
import cv2
import numpy as np
from torchvision import transforms
from main.model import get_pose_net
from main.config import cfg

# Load the model and set it in evaluation mode
posenet = get_pose_net(cfg, is_train=False, joint_num=17)
checkpoint = torch.load("C:/Users/thoma/OneDrive/Documents/EPFL/MA2/Deep Learning for Autonomous Vehicles/DLAV/3DMPPE_POSENET_RELEASE/snapshot_24.pth.tar", map_location=torch.device('cpu'))

model_state = checkpoint['network']
optimizer_state = checkpoint['optimizer']

posenet.load_state_dict(model_state, strict=False)

posenet.eval()

# Load and preprocess the test image
image = cv2.imread("C:/Users/thoma/OneDrive/Documents/EPFL/MA2/Deep Learning for Autonomous Vehicles/DLAV/3DMPPE_POSENET_RELEASE/sample2.jpg")

# Resize the image to the desired input size
resized_image = cv2.resize(image, (256, 256))

# Convert the image to float and convert BGR to RGB
preprocessed_image = resized_image.astype(np.float32) / 255.0
preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)

# Define the COCO normalization parameters
coco_mean = torch.tensor([0.485, 0.456, 0.406])
coco_std = torch.tensor([0.229, 0.224, 0.225])

# Apply COCO normalization
normalize = transforms.Normalize(mean=coco_mean, std=coco_std)
preprocessed_image = normalize(torch.from_numpy(preprocessed_image).permute(2, 0, 1))

# Add a batch dimension
preprocessed_image = preprocessed_image.unsqueeze(0)

# Move the model and input to CPU
posenet = posenet.to(torch.device('cpu'))
preprocessed_image = preprocessed_image.to(torch.device('cpu'))

# Perform inference
with torch.no_grad():
    output = posenet(preprocessed_image)

print(output)

output = output.view(-1, output.shape[-1]).numpy()
# Save the array as a text file
np.savetxt('3d_joints.txt', output, delimiter=' ', fmt='%s')

# Process the output and use it as needed
# ...
