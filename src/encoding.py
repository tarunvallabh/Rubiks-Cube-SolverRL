import numpy as np
from cube import Cube


# encoding.py
import torch


def encode_full(faces):
    """Convert cube state to one-hot encoded input tensor"""
    # Initialize tensor
    state = torch.zeros(6 * 4 * 6)  # Add device placement

    # Face order: Up, Down, Front, Back, Left, Right
    face_names = ["Up", "Down", "Front", "Back", "Left", "Right"]

    idx = 0
    for face_name in face_names:
        for sticker in faces[face_name]:
            # One-hot encode each sticker color
            state[idx * 6 + sticker] = 1
            idx += 1

    return state


def decode_full(encoded_state):
    """Convert one-hot encoded state back to cube faces
    Args:
        encoded_state: torch/numpy array of shape (144,) containing one-hot encoded state
    Returns:
        dict with faces["Up","Down","Front","Back","Left","Right"] -> list of 4 colors each
    """
    # Initialize faces dictionary
    faces = {"Up": [], "Down": [], "Front": [], "Back": [], "Left": [], "Right": []}

    # Face order matches encoding order
    face_names = ["Up", "Down", "Front", "Back", "Left", "Right"]

    # For each face
    for face_idx, face_name in enumerate(face_names):
        # For each sticker on the face
        for sticker in range(4):
            # Calculate where this sticker's one-hot encoding starts
            # Each sticker takes 6 positions (one for each possible color)
            start_idx = (face_idx * 4 + sticker) * 6

            # Get the next 6 positions (the one-hot encoding for this sticker)
            one_hot = encoded_state[start_idx : start_idx + 6]

            # Find which position is 1 - that's our color
            color = np.argmax(one_hot)

            # Add to our face
            faces[face_name].append(color)

    return faces


# def demonstrate_encodings():
#     cube = Cube(visualize=True)

#     print("SOLVED CUBE:")
#     print("\nFirst few positions of full encoding:")
#     full = encode_full(cube.faces)
#     # Look at first 12 numbers (2 stickers)
#     print(full[:24].reshape(4, 6))  # Reshape to see the one-hot encoding clearly
#     full_faces = decode_full(full)
#     print("\nDecoded faces:")
#     for face, colors in full_faces.items():
#         print(f"{face}: {colors}")

#     # print("\nFirst few positions of corner encoding:")
#     # corner = one_hot_encode_corners(cube)
#     # Look at first 12 numbers (2 stickers)
#     # print(corner.reshape(14, 6))

#     print("\nNow let's make a move...")
#     cube.execute_move("R")  # Make a right face turn

#     print("\nAfter RIGHT move:")
#     print("Full encoding first few positions:")
#     full_after = encode_full(cube.faces)
#     full_after_faces = decode_full(full_after)
#     print(full_after[:24].reshape(4, 6))
#     for face, colors in full_after_faces.items():
#         print(f"{face}: {colors}")


# demonstrate_encodings()
