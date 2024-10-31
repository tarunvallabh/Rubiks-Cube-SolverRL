import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


class Cube:
    def __init__(self):
        self.cube_colors = {
            "white": 0,
            "yellow": 1,
            "green": 2,
            "blue": 3,
            "red": 4,
            "orange": 5,
        }

        self.color_list = [
            (1, 1, 1),  # 0: White
            (1, 1, 0),  # 1: Yellow
            (0, 1, 0),  # 2: Green
            (0, 0, 1),  # 3: Blue
            (1, 0, 0),  # 4: Red
            (1, 0.5, 0),  # 5: Orange
        ]

        self.faces = {
            "Up": [self.cube_colors["white"] for _ in range(4)],
            "Down": [self.cube_colors["yellow"] for _ in range(4)],
            "Front": [self.cube_colors["green"] for _ in range(4)],
            "Back": [self.cube_colors["blue"] for _ in range(4)],
            "Left": [self.cube_colors["red"] for _ in range(4)],
            "Right": [self.cube_colors["orange"] for _ in range(4)],
        }

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")
        plt.ion()  # Turn on interactive mode

    def is_solved(self):
        """Check if the cube is solved"""
        return all(len(set(face)) == 1 for face in self.faces.values())

    def rotate_face_clockwise(self, face):
        face[0], face[1], face[2], face[3] = face[2], face[0], face[3], face[1]

    def rotate_face_counterclockwise(self, face):
        face[0], face[1], face[2], face[3] = face[1], face[3], face[0], face[2]

    def move_left(self):
        self.rotate_face_clockwise(self.faces["Left"])

        # store the edge pieces
        up_edge = [self.faces["Up"][0], self.faces["Up"][2]]
        front_edge = [self.faces["Front"][0], self.faces["Front"][2]]
        down_edge = [self.faces["Down"][0], self.faces["Down"][2]]
        back_edge = [self.faces["Back"][0], self.faces["Back"][2]]

        # update the edge pieces
        # up adjacent to front
        self.faces["Front"][0], self.faces["Front"][2] = up_edge[0], up_edge[1]
        # front adjacent to down
        self.faces["Down"][0], self.faces["Down"][2] = front_edge[0], front_edge[1]
        # down adjacent to back
        self.faces["Back"][0], self.faces["Back"][2] = down_edge[0], down_edge[1]
        # back adjacent to up
        self.faces["Up"][0], self.faces["Up"][2] = back_edge[0], back_edge[1]

        self.plot_cube()

        # no need to return self.faces since it is passed by reference

    def move_right(self):
        self.rotate_face_clockwise(self.faces["Right"])

        # store the edge pieces
        up_edge = [self.faces["Up"][1], self.faces["Up"][3]]
        front_edge = [self.faces["Front"][1], self.faces["Front"][3]]
        down_edge = [self.faces["Down"][1], self.faces["Down"][3]]
        back_edge = [self.faces["Back"][1], self.faces["Back"][3]]

        # update the edge pieces
        # up adjacent to back
        self.faces["Back"][1], self.faces["Back"][3] = up_edge[0], up_edge[1]
        # back adjacent to down
        self.faces["Down"][1], self.faces["Down"][3] = back_edge[0], back_edge[1]
        # down adjacent to front
        self.faces["Front"][1], self.faces["Front"][3] = down_edge[0], down_edge[1]
        # front adjacent to up
        self.faces["Up"][1], self.faces["Up"][3] = front_edge[0], front_edge[1]

        self.plot_cube()

        # no need to return self.faces since it is passed by reference

    def move_up(self):
        self.rotate_face_clockwise(self.faces["Up"])

        # Store the edge pieces
        front_edge = [self.faces["Front"][0], self.faces["Front"][1]]
        right_edge = [self.faces["Right"][0], self.faces["Right"][1]]
        back_edge = [self.faces["Back"][0], self.faces["Back"][1]]
        left_edge = [self.faces["Left"][0], self.faces["Left"][1]]

        # Update the edge pieces in clockwise order
        # Front adjacent to Left
        # Front's top edge goes to Right
        self.faces["Right"][0], self.faces["Right"][1] = front_edge[0], front_edge[1]
        # Right's top edge goes to Back
        self.faces["Back"][0], self.faces["Back"][1] = right_edge[0], right_edge[1]
        # Back's top edge goes to Left
        self.faces["Left"][0], self.faces["Left"][1] = back_edge[0], back_edge[1]
        # Left's top edge goes to Front
        self.faces["Front"][0], self.faces["Front"][1] = left_edge[0], left_edge[1]

        self.plot_cube()

    def move_down(self):
        self.rotate_face_clockwise(self.faces["Down"])

        # Store the edge pieces
        front_edge = [self.faces["Front"][2], self.faces["Front"][3]]
        right_edge = [self.faces["Right"][2], self.faces["Right"][3]]
        back_edge = [self.faces["Back"][2], self.faces["Back"][3]]
        left_edge = [self.faces["Left"][2], self.faces["Left"][3]]

        # Update the edge pieces in clockwise order
        # Front adjacent to Right
        self.faces["Right"][2], self.faces["Right"][3] = front_edge[0], front_edge[1]
        # Right adjacent to Back
        self.faces["Back"][2], self.faces["Back"][3] = right_edge[0], right_edge[1]
        # Back adjacent to Left
        self.faces["Left"][2], self.faces["Left"][3] = back_edge[0], back_edge[1]
        # Left adjacent to Front
        self.faces["Front"][2], self.faces["Front"][3] = left_edge[0], left_edge[1]

        self.plot_cube()

    def move_front(self):
        self.rotate_face_clockwise(self.faces["Front"])

        # Store the edge pieces
        up_edge = [self.faces["Up"][2], self.faces["Up"][3]]
        right_edge = [self.faces["Right"][0], self.faces["Right"][2]]
        down_edge = [self.faces["Down"][0], self.faces["Down"][1]]
        left_edge = [self.faces["Left"][1], self.faces["Left"][3]]

        # Update the edge pieces in clockwise order
        # Up adjacent to Left
        # Up's bottom edge goes to Right's left edge
        self.faces["Right"][0], self.faces["Right"][2] = up_edge[0], up_edge[1]
        # Right's left edge goes to Down's top edge
        self.faces["Down"][0], self.faces["Down"][1] = right_edge[0], right_edge[1]
        # Down's top edge goes to Left's right edge
        self.faces["Left"][1], self.faces["Left"][3] = down_edge[0], down_edge[1]
        # Left's right edge goes to Up's bottom edge
        self.faces["Up"][2], self.faces["Up"][3] = left_edge[0], left_edge[1]

        self.plot_cube()

    def move_back(self):
        self.rotate_face_clockwise(self.faces["Back"])

        # Store edge pieces
        up_edge = [self.faces["Up"][0], self.faces["Up"][1]]  # Top edge of Up face
        left_edge = [
            self.faces["Left"][0],
            self.faces["Left"][2],
        ]  # Left edge of Left face
        down_edge = [
            self.faces["Down"][2],
            self.faces["Down"][3],
        ]  # Bottom edge of Down face
        right_edge = [
            self.faces["Right"][1],
            self.faces["Right"][3],
        ]  # Right edge of Right face

        # When Back rotates clockwise:
        # Up's top edge goes to Left's left edge
        self.faces["Left"][0], self.faces["Left"][2] = up_edge[0], up_edge[1]
        # Left's left edge goes to Down's bottom edge
        self.faces["Down"][2], self.faces["Down"][3] = left_edge[0], left_edge[1]
        # Down's bottom edge goes to Right's right edge
        self.faces["Right"][1], self.faces["Right"][3] = down_edge[0], down_edge[1]
        # Right's right edge goes to Up's top edge
        self.faces["Up"][0], self.faces["Up"][1] = right_edge[0], right_edge[1]

        self.plot_cube()

    def __str__(self):
        """Visual representation of the cube in a 2D net format."""
        colors_map = {
            0: "W",  # White
            1: "Y",  # Yellow
            2: "G",  # Green
            3: "B",  # Blue
            4: "R",  # Red
            5: "O",  # Orange
        }

        # Helper function to get the 2x2 grid for a face
        def get_face_grid(face_name):
            colors = self.faces[face_name]
            grid = [
                [colors_map[colors[0]], colors_map[colors[1]]],
                [colors_map[colors[2]], colors_map[colors[3]]],
            ]
            return grid

        # Get grids for each face
        up = get_face_grid("Up")
        down = get_face_grid("Down")
        left = get_face_grid("Left")
        right = get_face_grid("Right")
        front = get_face_grid("Front")
        back = get_face_grid("Back")

        # Build the output lines
        lines = []

        # First, print the Up face centered
        blank = " " * 6  # Adjust spaces as needed
        lines.append(blank + " ".join(up[0]))
        lines.append(blank + " ".join(up[1]))
        lines.append("")  # Empty line for spacing

        # Then, print Left, Front, Right, Back faces in a row
        for row in range(2):
            line = []
            # Left face
            line.extend(left[row])
            line.append(" ")  # Space between faces
            # Front face
            line.extend(front[row])
            line.append(" ")
            # Right face
            line.extend(right[row])
            line.append(" ")
            # Back face
            line.extend(back[row])
            lines.append(" ".join(line))
        lines.append("")  # Empty line for spacing

        # Then, print the Down face centered
        lines.append(blank + " ".join(down[0]))
        lines.append(blank + " ".join(down[1]))

        return "\n".join(lines)

    def plot_cube(self):
        self.ax.clear()  # Clear the axes to redraw

        # Define sticker size
        sticker_size = 1  # Adjust as needed

        # Define the positions for each face
        positions = {
            "Front": (0, 0, 1),  # Front face at positive Z-axis
            "Back": (0, 0, -1),  # Back face at negative Z-axis
            "Up": (0, 1, 0),  # Up face at positive Y-axis (remains the same)
            "Down": (0, -1, 0),  # Down face at negative Y-axis
            "Left": (-1, 0, 0),  # Left face at negative X-axis
            "Right": (1, 0, 0),  # Right face at positive X-axis
        }

        # Define the orientations for each face
        orientations = {
            "Front": (0, 0),  # No rotation needed
            "Back": (0, np.pi),  # Rotate 180 degrees around Y-axis
            "Up": (-np.pi / 2, 0),  # Rotate -90 degrees around X-axis
            "Down": (np.pi / 2, 0),  # Rotate 90 degrees around X-axis
            "Left": (0, np.pi / 2),  # Rotate 90 degrees around Y-axis
            "Right": (0, -np.pi / 2),  # Rotate -90 degrees around Y-axis
        }

        for face in self.faces:
            self.plot_face(
                self.ax, face, positions[face], orientations[face], sticker_size
            )

        # Set the aspect ratio to 'auto' to prevent distortion
        self.ax.set_box_aspect([1, 1, 1])

        # Hide the axes
        self.ax.axis("off")

        # Set the viewing angle so that the 'Front' face is facing the viewer
        self.ax.view_init(elev=90, azim=-90)

        plt.draw()
        plt.pause(1)  # Pause to allow the plot to update

    def plot_face(self, ax, face_name, position, orientation, sticker_size):
        # Get the colors for the face's stickers
        face_colors = [
            self.color_list[color_idx] for color_idx in self.faces[face_name]
        ]

        # Define the sticker positions on the face
        sticker_positions = [
            (-0.5, 0.5),  # Top-left
            (0.5, 0.5),  # Top-right
            (-0.5, -0.5),  # Bottom-left
            (0.5, -0.5),  # Bottom-right
        ]

        # Create a square for each sticker
        for idx, (x, y) in enumerate(sticker_positions):
            # Define the square in 2D
            s = sticker_size / 2
            square = np.array(
                [[x - s, y - s], [x - s, y + s], [x + s, y + s], [x + s, y - s]]
            )

            # Rotate the square according to the face orientation
            theta, phi = orientation
            # Rotation matrices
            R_theta = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)],
                ]
            )
            R_phi = np.array(
                [
                    [np.cos(phi), 0, np.sin(phi)],
                    [0, 1, 0],
                    [-np.sin(phi), 0, np.cos(phi)],
                ]
            )
            R = R_theta @ R_phi

            # Convert 2D square to 3D
            square_3d = np.array([[x, y, 0] for x, y in square])

            # Rotate and translate the square
            square_rotated = np.dot(square_3d, R.T) + np.array(position)

            # Create the polygon and add it to the plot
            poly = Poly3DCollection(
                [square_rotated], facecolors=face_colors[idx], edgecolors="black"
            )
            ax.add_collection3d(poly)


# ... [Rest of the code remains unchanged] ...

if __name__ == "__main__":
    cube = Cube()
    print("Initial cube state:")
    print(cube)
    print("\nIs solved:", cube.is_solved())

    # Initial plot
    cube.plot_cube()

    print("\nPerforming some moves...")
    # cube.move_right()
    # cube.move_up()
    # cube.move_front()

    print("\nCube state after moves:")
    print(cube)
    print("\nIs solved:", cube.is_solved())

    # Keep the plot open
    plt.ioff()  # Turn off interactive mode
    plt.show()
