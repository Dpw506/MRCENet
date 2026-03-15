import cv2
import numpy as np
import os

def process_images(y_image_dir, rgb_image_dir, output_dir):
    # Get the list of Y channel image and RGB image filenames
    y_image_files = [f for f in os.listdir(y_image_dir) if f.endswith(('.jpg', '.png'))]
    rgb_image_files = [f for f in os.listdir(rgb_image_dir) if f.endswith(('.jpg', '.png'))]

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for y_file, rgb_file in zip(y_image_files, rgb_image_files):
        # Open the Y channel image and RGB image
        y_image_path = os.path.join(y_image_dir, y_file)
        rgb_image_path = os.path.join(rgb_image_dir, rgb_file)

        Y_image = cv2.imread(y_image_path, cv2.IMREAD_GRAYSCALE)  # Y channel image
        original_rgb_image = cv2.imread(rgb_image_path, cv2.IMREAD_COLOR)  # Original RGB image

        # Resize the Y channel image to match the size of the RGB image
        Y_image_resized = cv2.resize(Y_image, (original_rgb_image.shape[1], original_rgb_image.shape[0]))

        # Extract Cb and Cr channels from the RGB image
        YUV_image = cv2.cvtColor(original_rgb_image, cv2.COLOR_BGR2YCrCb)
        Cb_array = YUV_image[:, :, 1]
        Cr_array = YUV_image[:, :, 2]

        # Create an empty YUV image array
        YUV_array = np.zeros((original_rgb_image.shape[0], original_rgb_image.shape[1], 3), dtype=np.uint8)

        # Merge Y, Cb, and Cr channels into the YUV image array
        YUV_array[:, :, 0] = Y_image_resized  # Y channel
        YUV_array[:, :, 1] = Cb_array  # Cb channel
        YUV_array[:, :, 2] = Cr_array  # Cr channel

        # Convert the YUV image back to RGB
        RGB_result_image = cv2.cvtColor(YUV_array, cv2.COLOR_YCrCb2BGR)

        # Save the final RGB image
        output_path = os.path.join(output_dir, y_file)
        cv2.imwrite(output_path, RGB_result_image)

        print(f"Processed {y_file} and {rgb_file} -> {output_path}")

# Set the input and output directories
y_image_dir = './output/SPECT-MRI/Net_v25_11010_n9'
rgb_image_dir = 'data/SPECT-MRI/test/SPECT'
output_dir = './output/SPECT-MRI/Net_v25_11010_n9_color'

# y_image_dir = './output/Whu_Data_82/Net_v22_11010_sample_11'
# rgb_image_dir = 'data/Sample_11/PET_11_nii2png'
# output_dir = './output/Whu_Data_82/Net_v22_11010_sample_11_color'

# y_image_dir = './output/PET-MRI/Net_v25_11010_n9'
# rgb_image_dir = 'data/PET-MRI/test/PET'
# output_dir = './output/PET-MRI/Net_v25_11010_n9_color'

# Batch process images
process_images(y_image_dir, rgb_image_dir, output_dir)
