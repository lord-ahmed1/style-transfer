import os
import random

def delete_random_images_keep_n(folder_path, keep_n=1000):
    # List all image files (common image extensions)
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    images = [f for f in os.listdir(folder_path) 
              if os.path.splitext(f)[1].lower() in valid_exts]
    
    total_images = len(images)
    print(f"Total images found: {total_images}")

    if total_images <= keep_n:
        print(f"No images deleted because there are only {total_images} images.")
        return

    # Number of images to delete
    num_to_delete = total_images - keep_n
    print(f"Deleting {num_to_delete} images at random...")

    # Randomly select images to delete
    images_to_delete = random.sample(images, num_to_delete)

    for img in images_to_delete:
        img_path = os.path.join(folder_path, img)
        try:
            os.remove(img_path)
        except Exception as e:
            print(f"Failed to delete {img_path}: {e}")

    print(f"Deleted {num_to_delete} images, kept {keep_n} images.")

# Example usage
folder = "style"
delete_random_images_keep_n(folder, keep_n=1000)
