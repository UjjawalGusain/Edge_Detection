from PIL import Image
import numpy as np

def turn_img_to_pixels(image):
    """
    Converts PIL image into 2D Numpy array 
    
    :param image: A PIL image
    """
    pixel_array = np.array(image, dtype=np.uint8)
    return pixel_array

def show_picture(pixel_array, save=False):
    """
    Shows and saves the picture 
    
    :param pixel_array: Input 2D array (image) to be viewed.
    :param save: bool to imply if image be saved.
    """
    image = Image.fromarray(pixel_array, mode='L')
    if save:
        image.save('restored_image.jpg')
    image.show()

def convolution_operation(pixel_array, filter):
    """
    Performs a 3x3 convolution operation on a pixel_array using the provided filter.
    
    :param pixel_array: Input 2D array (image) to be convolved.
    :param filter: 3x3 kernel/filter for the convolution operation.
    :return: Convolved output array (same size as input pixel_array).
    """
    assert len(pixel_array.shape) == 2, "Input image must be grayscale"
    n, m = pixel_array.shape
    
    # Padding the image
    padded_array = np.pad(pixel_array, pad_width=1, mode='constant')

    # Performing convolution using slicing and vectorizing
    convolved_array = np.zeros_like(pixel_array, dtype=np.float32)
    for i in range(1, n+1):
        for j in range(1, m+1):
            roi = padded_array[i-1:i+2, j-1:j+2]
            convolved_value = np.sum(roi * filter)
            convolved_array[i-1, j-1] = np.clip(convolved_value, 0, 255)

    return convolved_array.astype(np.uint8)

if __name__ == "__main__":
    image_path = 'testSample/img_eye.jpg'
    image = Image.open(image_path).convert('L')  # Converting to grayscale
    
    pixel_array = turn_img_to_pixels(image)

    top_filter = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]])
    
    left_filter = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
    
    bottom_filter = np.array([[1, 2, 1],
                              [0, 0, 0],
                              [-1, -2, -1]])
    
    right_filter = np.array([[1, 0, -1],
                             [2, 0, -2],
                             [1, 0, -1]])

    # Performing convolution operations in every direction
    top_array_image = convolution_operation(pixel_array, top_filter)
    left_array_image = convolution_operation(pixel_array, left_filter)
    bottom_array_image = convolution_operation(pixel_array, bottom_filter)
    right_array_image = convolution_operation(pixel_array, right_filter)

    combined_image = top_array_image + left_array_image + bottom_array_image + right_array_image
    combined_image_clipped = np.clip(combined_image, 0, 255)
    show_picture(combined_image_clipped, save=True)
