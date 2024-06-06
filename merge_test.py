import glob

import cv2
import numpy as np
import tqdm

# Path to the folder containing the time-lapse images
image_folder = "/home/davidyz/Pictures/圣母百花日落/JPG/"

# Load all images in the folder
file_paths = sorted(glob.glob(f"{image_folder}/*.JPG"), reverse=True)[::5]
images = [cv2.imread(file) for file in file_paths]

# Ensure images are loaded
if not images:
    raise ValueError("No images found in the specified folder")


# Get the dimensions of the images
height, width, _ = images[0].shape

# Number of slices
num_slices = len(images)
slice_width = width // num_slices

# Initialize the merged image
merged_image = np.zeros_like(images[0], dtype=np.float32)

# Set the width of the transition region
transition_width = slice_width

# Blend each image slice with overlapping regions
for i in tqdm.tqdm(range(num_slices)):
    start_x = i * slice_width
    end_x = start_x + slice_width

    if i > 0:
        prev_start_x = start_x - transition_width
        prev_end_x = start_x

        mask: np.ndarray = np.linspace(0, 1, transition_width, dtype=np.float32)
        mask = np.tile(mask[np.newaxis, :], (height, 1))
        mask = np.dstack([mask] * 3)
        inv_mask = 1 - mask

        merged_image[:, prev_start_x:prev_end_x] = (
            merged_image[:, prev_start_x:prev_end_x] * inv_mask
            + images[i][:, prev_start_x:prev_end_x] * mask
        )

    merged_image[:, start_x:end_x] = images[i][:, start_x:end_x]

# Convert the merged image back to uint8
merged_image = cv2.convertScaleAbs(merged_image)
# Save the merged image
output_path = "merged_image.jpg"
cv2.imwrite(output_path, merged_image)
