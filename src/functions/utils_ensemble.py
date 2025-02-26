# Function to discretize an array with a sliding window
def discretize(arr: np.ndarray, window_size: int) -> np.ndarray:
    """
    Discretizes an image (2D array) by averaging values in a sliding window.

    :param arr: Input 2D numpy array.
    :param window_size: Size of the window used to average the values.
    :return: 2D numpy array with averaged values.
    """
    H, W = arr.shape
    output = np.zeros((H, W), dtype=np.float32)

    # Iterate over the array in steps of window_size
    for i in range(0, H, window_size):
        y_end = min(i + window_size, H)
        for j in range(0, W, window_size):
            x_end = min(j + window_size, W)
            output[i:y_end, j:x_end] = np.mean(arr[i:y_end, j:x_end])

    return output


# Function to add text to an image
def add_text_to_image(img: np.ndarray, text: str, space: int = 10,
                      color: tuple = (255, 255, 255), thickness: int = 2,
                      scale: float = 0.5) -> np.ndarray:
    """
    Adds centered text to an image.

    :param img: Input image (numpy array).
    :param text: The text to be added.
    :param space: Space from the top of the image for the text.
    :param color: Color of the text in BGR format (default is white).
    :param thickness: Thickness of the text (default is 2).
    :param scale: Scale of the text (default is 0.5).
    :return: The image with added text.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    text_width, text_height = text_size

    # Calculate position to center text horizontally and place it vertically
    height, width, _ = img.shape
    x = (width - text_width) // 2
    y = text_height + space if space else text_height + 10  # Top margin

    # Add the text to the image
    cv2.putText(img, text, (x, y), font, scale, color, thickness)

    return img


# Function to normalize an array to a range between 0 and 1
def normalize(arr: np.ndarray) -> np.ndarray:
    """
    Normalizes the array values to the range [0, 1].

    :param arr: Input 2D or 1D numpy array.
    :return: Normalized numpy array.
    """
    if np.any(arr):
        arr = np.maximum(0, arr)  # Ensure non-negative values
        arr = arr - arr.min()  # Shift values to start from 0
        return arr / (arr.max() - arr.min())  # Scale to [0, 1]
    return arr  # Return the input array if it is empty or all zeros


# Utility function to apply threshold and normalize values
def threshold(img: np.ndarray, res: list, thres: float, binary: bool = False, size: int = 14) -> np.ndarray:
    """
    Applies a threshold to the ensemble of results and returns a combined output.

    :param img: The input image (not used directly in the function).
    :param res: List of result arrays to combine.
    :param thres: Threshold value for inclusion.
    :param binary: Whether to return a binary mask (default: False).
    :param size: Size parameter (not used directly).
    :return: Combined output array, either binary or normalized.
    """
    H, W = res[0].shape
    output = np.zeros((H, W), dtype=float)
    mask = np.zeros((H, W), dtype=float)

    # Apply thresholding across each result array
    for arr in res:
        normalized_arr = normalize(arr)
        mask += np.where(normalized_arr >= thres, 1, 0)
        output += np.where(normalized_arr >= thres, normalized_arr, 0)

    output = output / len(res)
    majority = np.where(mask >= len(res) - 1, 1, 0)

    return majority if binary else normalize(output) * majority


# Utility function to average the results of multiple arrays
def average(res: list, binary: bool = False, size: int = 14) -> np.ndarray:
    """
    Averages the results across multiple arrays.

    :param res: List of result arrays to average.
    :param binary: Whether to return a binary mask (default: False).
    :param size: Size parameter (not used directly).
    :return: Averaged result array, either binary or normalized.
    """
    H, W = res[0].shape
    output = np.zeros((H, W), dtype=float)
    mask = np.zeros((H, W), dtype=float)

    # Average the values across the result arrays
    for arr in res:
        normalized_arr = normalize(arr)
        output += np.where(normalized_arr > 0, normalized_arr, 0)
        mask += np.where(normalized_arr > 0, 1, 0)

    output = output / len(res)
    majority = np.where(mask >= len(res) / 2, 1, 0)

    if binary:
        return majority
    else:
        output = output * majority
        return np.where(output > 0.5, output, 0)


# Utility function to find intersection of multiple result arrays
def intersection(res: list, binary: bool = False, size: int = 14) -> np.ndarray:
    """
    Returns the intersection of results from multiple arrays.

    :param res: List of result arrays to combine.
    :param binary: Whether to return a binary mask (default: False).
    :param size: Size parameter (not used directly).
    :return: Intersection result array, either binary or normalized.
    """
    H, W = res[0].shape
    output = np.zeros((H, W), dtype=float)
    mask = np.zeros((H, W), dtype=float)

    # Calculate intersection by considering values above a certain threshold
    for arr in res:
        normalized_arr = normalize(arr)
        output += np.where(normalized_arr > 0.2, normalized_arr, 0)
        mask += np.where(normalized_arr > 0.2, 1, 0)

    output = output / len(res)
    majority = np.where(mask == len(res), 1, 0)

    if binary:
        return majority
    else:
        output = output * majority
        return np.where(output, output, 0)
