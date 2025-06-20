from datasets import load_dataset
import os
import pandas as pd
from PIL import Image
import io

# Define your project's data directory
# Assuming your script is in `scripts/` and data is in `data/raw/`
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
raw_data_dir = os.path.join(project_root, 'data', 'raw', 'Kvasir-VQA')
os.makedirs(raw_data_dir, exist_ok=True)
images_output_dir = os.path.join(raw_data_dir, 'images')
os.makedirs(images_output_dir, exist_ok=True)
metadata_output_path = os.path.join(raw_data_dir, 'metadata.csv')

print(f"Downloading and preparing Kvasir-VQA dataset to: {raw_data_dir}")

# Load the dataset
# The default split contains 58.8k rows of image+text data.
# The 'image' column in the dataset is of type Image (PIL.Image.Image)
dataset = load_dataset("SimulaMet-HOST/Kvasir-VQA", split="raw")

# The dataset typically consists of:
# 'image': PIL Image object
# 'source': string (e.g., 'HyperKvasir')
# 'question': string
# 'answer': string
# 'img_id': string (unique ID for the image)

# Convert dataset to a Pandas DataFrame for easier manipulation of metadata
# And extract image data to files
data_list = []
for i, item in enumerate(dataset):
    image = item['image']
    img_id = item['img_id']
    image_filename = f"{img_id}.jpg" # You can choose .png if preferred
    image_path = os.path.join(images_output_dir, image_filename)

    # Save the image
    # PIL Image objects can be directly saved
    image.save(image_path)

    # Store metadata without the image object itself
    metadata_item = {
        'source': item['source'],
        'question': item['question'],
        'answer': item['answer'],
        'img_id': img_id,
        'image_filepath': os.path.relpath(image_path, raw_data_dir) # Relative path for portability
    }
    data_list.append(metadata_item)

    if (i + 1) % 1000 == 0:
        print(f"Processed {i + 1} examples...")

# Save the metadata to a CSV file
metadata_df = pd.DataFrame(data_list)
metadata_df.to_csv(metadata_output_path, index=False)

print(f"\nKvasir-VQA dataset downloaded and processed to: {raw_data_dir}")
print(f"Total images saved: {len(metadata_df)}")
print(f"Metadata saved to: {metadata_output_path}")

# Example of how to access it later
# # Read metadata
# loaded_metadata = pd.read_csv(metadata_output_path)
# # Construct full image path
# first_image_path = os.path.join(raw_data_dir, loaded_metadata['image_filepath'].iloc[0])
# # Load image
# loaded_image = Image.open(first_image_path)