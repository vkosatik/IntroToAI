import os
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
import re

# Initialize the model and processor
print("Initializing model and processor...")
model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")
processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

prompt = "<grounding>An image of"


# Define natural sorting function to sort filenames numerically if needed
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


# Load multiple images from the directory and sort them naturally
image_dir = '/Users/vojtechkosatik/PycharmProjects/IntroToAILabs/Lab6/extracted/764079a60a57e1baf86fdfcfc4df10cb909f335767fa5b08cca690f6fb548f9e/test'
image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith(('png', 'jpg', 'jpeg'))]

# Sort the image files alphabetically or numerically
image_files.sort(key=natural_sort_key)  # This will handle numeric sorting if filenames contain numbers

# Process images in smaller batches
batch_size = 1
for i in range(0, len(image_files), batch_size):
    batch_files = image_files[i:i + batch_size]

    # Display the image filenames being processed to verify order
    for image_file in batch_files:
        print(f"Processing image: {image_file}")

    # Load the images and resize them
    images = [Image.open(image_file).convert("RGB") for image_file in batch_files]
    images = [image.resize((224, 224)) for image in images]  # Resize images to 224x224 for the model

    # Replicate the prompt for each image
    prompts = [prompt] * len(images)

    # Process the inputs
    print(f"Processing batch {i // batch_size + 1}...")
    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)

    # Generate text for each image
    print("Generating text...")
    generated_ids = model.generate(
        pixel_values=inputs["pixel_values"],
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_embeds=None,
        image_embeds_position_mask=inputs["image_embeds_position_mask"],
        use_cache=True,
        max_new_tokens=128,
    )

    # Decode the generated text
    print("Decoding generated text...")
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

    # Clean up the generated texts
    cleaned_texts = []
    for text in generated_texts:
        # Extract the part starting from "An image of"
        start_index = text.find("An image of")
        if start_index != -1:
            # Extract the relevant portion
            text = text[start_index:]
            # Remove any object tags, numbers, and unnecessary phrases
            text = re.sub(r"<.*?>", "", text)  # Remove tags like <grounding>, <object>, etc.
            text = re.sub(r"\d{6,}", "", text)  # Remove any long numeric strings
            text = text.replace("</delimiter_of_multi_objects/", "").strip()  # Remove any remaining unnecessary phrases
            cleaned_texts.append(text)
        else:
            cleaned_texts.append("An image description could not be generated.")

    # Print the cleaned generated texts
    for j, text in enumerate(cleaned_texts):
        print(f"Generated Text for Image {i + j + 1}: {text}")