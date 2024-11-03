import numpy as np
from cube import Cube


def one_hot_encode_full(self, dtype: float = float) -> np.array:
    # Create a matrix of zeros: 24 rows (one for each sticker) × 6 columns (one for each possible color)
    result = np.zeros((24, 6), dtype=dtype)

    # List of faces in order we'll process them
    faces_order = ["Up", "Down", "Front", "Back", "Left", "Right"]

    # Go through each face
    for face_idx, face_name in enumerate(faces_order):
        face = self.faces[face_name]  # Get the face's stickers
        # Go through each sticker on this face
        for sticker_idx in range(4):
            color_idx = face[sticker_idx]  # Get the color number (0-5)
            # Put a 1 in the position for this sticker's color
            # face_idx * 4 + sticker_idx gives us position 0-23 for the sticker
            # color_idx tells us which column (color) gets the 1
            result[face_idx * 4 + sticker_idx, color_idx] = 1

    return result.flatten()  # Convert 24x6 matrix to 144-length vector


def demonstrate_encodings():
    cube = Cube()

    print("SOLVED CUBE:")
    print("\nFirst few positions of full encoding:")
    full = one_hot_encode_full(cube)
    # Look at first 12 numbers (2 stickers)
    print(full[:24].reshape(4, 6))  # Reshape to see the one-hot encoding clearly

    print("\nFirst few positions of corner encoding:")
    # corner = one_hot_encode_corners(cube)
    # Look at first 12 numbers (2 stickers)
    # print(corner.reshape(14, 6))

    print("\nNow let's make a move...")
    cube.execute_move("R")  # Make a right face turn

    print("\nAfter RIGHT move:")
    print("Full encoding first few positions:")
    full_after = one_hot_encode_full(cube)
    print(full_after[:24].reshape(4, 6))

    print("\nCorner encoding first few positions:")
    # corner_after = one_hot_encode_corners(cube)
    # print(corner_after.reshape(14, 6))


demonstrate_encodings()
