import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
import numpy as np
import re
import copy
from moves import Move


class Cube:
    def __init__(self, visualize=False):
        self.cube_colors = {
            "white": 0,
            "yellow": 1,
            "green": 2,
            "blue": 3,
            "red": 4,
            "orange": 5,
        }

        self.color_list = [
            "#FFFFFF",  # 0: White - pure white
            "#FFD500",  # 1: Yellow - vibrant yellow
            "#009B48",  # 2: Green - Rubik's green
            "#0046AD",  # 3: Blue - Rubik's blue
            "#B71234",  # 4: Red - Rubik's red
            "#FF5800",  # 5: Orange - Rubik's orange
        ]

        # Initialize solved cube state
        self.faces = {
            "Up": [self.cube_colors["white"] for _ in range(4)],
            "Down": [self.cube_colors["yellow"] for _ in range(4)],
            "Front": [self.cube_colors["green"] for _ in range(4)],
            "Back": [self.cube_colors["blue"] for _ in range(4)],
            "Left": [self.cube_colors["orange"] for _ in range(4)],
            "Right": [self.cube_colors["red"] for _ in range(4)],
        }

        self.visualize = visualize
        if self.visualize:
            self.color_list = [
                "#FFFFFF",  # White
                "#FFD500",  # Yellow
                "#009B48",  # Green
                "#0046AD",  # Blue
                "#B71234",  # Red
                "#FF5800",  # Orange
            ]
            self.fig = plt.figure(figsize=(10, 10))
            self.ax = self.fig.add_subplot(111, projection="3d")
            plt.ion()
        else:
            self.fig = None
            self.ax = None

        if self.visualize:
            self.plot_3d_cube()

    def plot_3d_cube(self):
        """Plot the cube in 3D using our previous visualization code"""
        if self.fig is None:
            self.fig = plt.figure(figsize=(10, 10))
            self.ax = self.fig.add_subplot(111, projection="3d")
            plt.ion()

        self.ax.cla()  # Clear the current plot

        def plot_colored_face(ax, vertices, color, alpha=1):
            vertices = np.array(vertices)
            ax.plot_surface(
                vertices[:, :, 0],
                vertices[:, :, 1],
                vertices[:, :, 2],
                color=color,
                alpha=alpha,
                edgecolor="black",
                linewidth=0.5,
                shade=False,
            )

        def plot_cube(ax, position, colors):
            x, y, z = position
            size = 1

            faces_vertices = {
                "front": np.array(
                    [
                        [[x, y, z], [x + size, y, z]],
                        [[x, y, z + size], [x + size, y, z + size]],
                    ]
                ),
                "back": np.array(
                    [
                        [[x, y + size, z], [x + size, y + size, z]],
                        [[x, y + size, z + size], [x + size, y + size, z + size]],
                    ]
                ),
                "left": np.array(
                    [
                        [[x, y + size, z], [x, y, z]],
                        [[x, y + size, z + size], [x, y, z + size]],
                    ]
                ),
                "right": np.array(
                    [
                        [[x + size, y, z], [x + size, y + size, z]],
                        [[x + size, y, z + size], [x + size, y + size, z + size]],
                    ]
                ),
                "top": np.array(
                    [
                        [[x, y, z + size], [x + size, y, z + size]],
                        [[x, y + size, z + size], [x + size, y + size, z + size]],
                    ]
                ),
                "bottom": np.array(
                    [
                        [[x, y, z], [x + size, y, z]],
                        [[x, y + size, z], [x + size, y + size, z]],
                    ]
                ),
            }

            for face, color_idx in colors.items():
                if color_idx is not None:
                    plot_colored_face(
                        ax, faces_vertices[face], self.color_list[color_idx]
                    )

        # Define cube configurations based on the current face states
        cube_configs = [
            # Front face, top row (left to right)
            {
                "pos": (0, 0, 1),
                "colors": {
                    "front": self.faces["Front"][0],
                    "left": self.faces["Left"][1],
                    "top": self.faces["Up"][2],  # Changed from [2]
                },
            },
            {
                "pos": (1, 0, 1),
                "colors": {
                    "front": self.faces["Front"][1],
                    "right": self.faces["Right"][0],
                    "top": self.faces["Up"][3],  # Changed from [3]
                },
            },
            # Front face, bottom row
            {
                "pos": (0, 0, 0),
                "colors": {
                    "front": self.faces["Front"][2],
                    "left": self.faces["Left"][3],
                    "bottom": self.faces["Down"][0],  # Changed from [0]
                },
            },
            {
                "pos": (1, 0, 0),
                "colors": {
                    "front": self.faces["Front"][3],
                    "right": self.faces["Right"][2],
                    "bottom": self.faces["Down"][1],  # Changed from [1]
                },
            },
            # Back face, top row
            {
                "pos": (0, 1, 1),
                "colors": {
                    "back": self.faces["Back"][1],
                    "left": self.faces["Left"][0],
                    "top": self.faces["Up"][0],
                },
            },
            {
                "pos": (1, 1, 1),
                "colors": {
                    "back": self.faces["Back"][0],
                    "right": self.faces["Right"][1],
                    "top": self.faces["Up"][1],
                },
            },
            # Back face, bottom row
            {
                "pos": (0, 1, 0),
                "colors": {
                    "back": self.faces["Back"][3],
                    "left": self.faces["Left"][2],
                    "bottom": self.faces["Down"][2],
                },
            },
            {
                "pos": (1, 1, 0),
                "colors": {
                    "back": self.faces["Back"][2],
                    "right": self.faces["Right"][3],
                    "bottom": self.faces["Down"][3],
                },
            },
        ]

        # Plot all cubes
        for config in cube_configs:
            plot_cube(self.ax, config["pos"], config["colors"])

        self.ax.set_box_aspect([2, 2, 2])
        # self.ax.set_xlabel("X")
        # self.ax.set_ylabel("Y")
        # self.ax.set_zlabel("Z")
        self.ax.set_xlim(0, 2)
        self.ax.set_ylim(0, 2)
        self.ax.set_zlim(0, 2)
        self.ax.set_axis_off()
        self.ax.view_init(elev=20, azim=-60)

        plt.title("2x2x2 Rubik's Cube")
        plt.draw()
        plt.pause(0.5)

    def rotate_face_clockwise(self, face):
        face[0], face[1], face[2], face[3] = face[2], face[0], face[3], face[1]

    def rotate_face_counterclockwise(self, face):
        face[0], face[1], face[2], face[3] = face[1], face[3], face[0], face[2]

    def move_left(self):
        """Rotate the left face clockwise (relative to FRU being fixed)."""
        self.rotate_face_clockwise(self.faces["Left"])

        # Use temporary variables a and b, exactly like the reference code
        a, b = (
            self.faces["Front"][0],
            self.faces["Front"][2],
        )  # a, b get Front's left edge
        self.faces["Front"][0], self.faces["Front"][2] = (
            self.faces["Up"][0],
            self.faces["Up"][2],
        )  # Front gets Up's left edge
        self.faces["Down"][0], self.faces["Down"][2], a, b = (
            a,
            b,
            self.faces["Down"][0],
            self.faces["Down"][2],
        )  # Down gets Front's old edge; a, b get Down's old edge
        self.faces["Back"][3], self.faces["Back"][1], a, b = (
            a,
            b,
            self.faces["Back"][3],
            self.faces["Back"][1],
        )  # Back gets Down's old edge; a, b get Back's old edge
        self.faces["Up"][0], self.faces["Up"][2] = a, b  # Up gets Back's old edge

        if self.visualize:
            self.plot_3d_cube()

    def move_down(self):
        """Rotate the Down face clockwise (relative to FRU being fixed)."""
        self.rotate_face_clockwise(self.faces["Down"])

        # Use temporary variables a and b, exactly like the reference code
        a, b = (
            self.faces["Front"][2],
            self.faces["Front"][3],
        )  # a, b get Front's bottom edge
        self.faces["Front"][2], self.faces["Front"][3] = (
            self.faces["Left"][2],
            self.faces["Left"][3],
        )  # Front gets Left's bottom edge
        self.faces["Right"][2], self.faces["Right"][3], a, b = (
            a,
            b,
            self.faces["Right"][2],
            self.faces["Right"][3],
        )  # Right gets Front's old edge; a, b get Right's old edge
        self.faces["Back"][2], self.faces["Back"][3], a, b = (
            a,
            b,
            self.faces["Back"][2],
            self.faces["Back"][3],
        )  # Back gets Right's old edge; a, b get Back's old edge
        self.faces["Left"][2], self.faces["Left"][3] = a, b  # Left gets Back's old edge

        if self.visualize:
            self.plot_3d_cube()

    def move_back(self):
        """Rotate the Back face clockwise (relative to FRU being fixed)."""
        self.rotate_face_clockwise(self.faces["Back"])

        # Use temporary variables a and b, exactly like the reference code
        a, b = (
            self.faces["Left"][0],
            self.faces["Left"][2],
        )  # a, b get Left's top/left edge
        self.faces["Left"][0], self.faces["Left"][2] = (
            self.faces["Up"][1],
            self.faces["Up"][0],
        )  # Left gets Up's top/right edge
        self.faces["Down"][2], self.faces["Down"][3], a, b = (
            a,
            b,
            self.faces["Down"][2],
            self.faces["Down"][3],
        )  # Down gets Left's old edge; a, b get Down's old edge
        self.faces["Right"][3], self.faces["Right"][1], a, b = (
            a,
            b,
            self.faces["Right"][3],
            self.faces["Right"][1],
        )  # Right gets Down's old edge; a, b get Right's old edge
        self.faces["Up"][1], self.faces["Up"][0] = a, b  # Up gets Right's old edge

        if self.visualize:
            self.plot_3d_cube()

    def move_left_prime(self):
        """Rotate the left face counter-clockwise."""
        self.rotate_face_counterclockwise(self.faces["Left"])

        a, b = self.faces["Front"][0], self.faces["Front"][2]
        self.faces["Front"][0], self.faces["Front"][2] = (
            self.faces["Down"][0],
            self.faces["Down"][2],
        )
        self.faces["Up"][0], self.faces["Up"][2], a, b = (
            a,
            b,
            self.faces["Up"][0],
            self.faces["Up"][2],
        )  # Up gets Front's old
        self.faces["Back"][3], self.faces["Back"][1], a, b = (
            a,
            b,
            self.faces["Back"][3],
            self.faces["Back"][1],
        )  # Back gets Up's old
        self.faces["Down"][0], self.faces["Down"][2] = a, b  # Down gets Back's old

        if self.visualize:
            self.plot_3d_cube()

    def move_down_prime(self):
        """Rotate the Down face counterclockwise."""
        self.rotate_face_counterclockwise(self.faces["Down"])

        a, b = self.faces["Front"][2], self.faces["Front"][3]
        self.faces["Front"][2], self.faces["Front"][3] = (
            self.faces["Right"][2],
            self.faces["Right"][3],
        )
        self.faces["Left"][2], self.faces["Left"][3], a, b = (
            a,
            b,
            self.faces["Left"][2],
            self.faces["Left"][3],
        )  # Left gets Front's old
        self.faces["Back"][2], self.faces["Back"][3], a, b = (
            a,
            b,
            self.faces["Back"][2],
            self.faces["Back"][3],
        )  # Back gets Left's old
        self.faces["Right"][2], self.faces["Right"][3] = a, b  # Right gets Back's old

        if self.visualize:
            self.plot_3d_cube()

    def move_back_prime(self):
        """Rotate the Back face counterclockwise."""
        self.rotate_face_counterclockwise(self.faces["Back"])

        a, b = self.faces["Left"][0], self.faces["Left"][2]
        self.faces["Left"][0], self.faces["Left"][2] = (
            self.faces["Down"][2],
            self.faces["Down"][3],
        )
        self.faces["Up"][1], self.faces["Up"][0], a, b = (
            a,
            b,
            self.faces["Up"][1],
            self.faces["Up"][0],
        )
        self.faces["Right"][3], self.faces["Right"][1], a, b = (
            a,
            b,
            self.faces["Right"][3],
            self.faces["Right"][1],
        )
        self.faces["Down"][2], self.faces["Down"][3] = a, b

        if self.visualize:
            self.plot_3d_cube()

    def execute_move(self, move):
        """Execute a single move on the cube."""
        match = re.match(r"([FBLRUD])('?)(\d*)", move.strip().upper())
        if not match:
            print(f"Invalid move: {move}")
            return

        face, prime, repetition = match.groups()
        repetition = int(repetition) if repetition else 1

        # Map the string input to Move enum
        move_map = {
            "L": Move.LEFT,
            "L'": Move.LEFT_PRIME,
            "D": Move.DOWN,
            "D'": Move.DOWN_PRIME,
            "B": Move.BACK,
            "B'": Move.BACK_PRIME,
        }

        move_str = face + prime
        if move_str in move_map:
            for _ in range(repetition):
                self.change_by(move_map[move_str])
        else:
            print(f"Invalid move: {move}")

    def execute_move_sequence(self, moves):
        """Execute a sequence of moves on the cube."""
        if "," in moves:
            move_list = moves.split(",")
            for move in move_list:
                self.execute_move(move)
        else:
            self.execute_move(moves)

    def change_by(self, move: Move):
        """Execute a single move based on the Move enum."""
        if move == Move.LEFT:
            self.move_left()
        elif move == Move.LEFT_PRIME:
            self.move_left_prime()
        elif move == Move.DOWN:
            self.move_down()
        elif move == Move.DOWN_PRIME:
            self.move_down_prime()
        elif move == Move.BACK:
            self.move_back()
        elif move == Move.BACK_PRIME:
            self.move_back_prime()
        else:
            assert False, "Invalid move"
        return self

    def print_raw_arrays(self):
        print("\nRaw face arrays:")
        for face_name, face_array in self.faces.items():
            print(f"{face_name}: {face_array}")
        print("\nColor mapping:")
        for color_name, color_num in self.cube_colors.items():
            print(f"{color_num}: {color_name}")

    def is_solved(self):
        """Check if the cube is solved"""
        return all(len(set(face)) == 1 for face in self.faces.values())

    def one_hot_encode(self):
        """One-hot encodes the cube state, using two faces per corner."""

        one_hot = np.zeros(84, dtype=np.float32)

        # Define the indices, mirroring the reference code's sticker selection
        indices = [
            (
                0,
                self.faces["Up"][2],
            ),  # U face, square index 2 (third element of the list)
            (1, self.faces["Front"][0]),  # F face, square index 0
            (2, self.faces["Up"][0]),  # U face, square index 0
            (3, self.faces["Left"][0]),  # L face, square index 0
            (4, self.faces["Up"][1]),  # U face, square index 1
            (5, self.faces["Back"][0]),  # B face, square index 0
            (6, self.faces["Down"][0]),  # D face, square index 0
            (7, self.faces["Front"][2]),  # F face, square index 2
            (8, self.faces["Down"][1]),  # D face, square index 1
            (9, self.faces["Right"][2]),  # R face, square index 2
            (10, self.faces["Down"][2]),  # D face, square index 2
            (11, self.faces["Left"][2]),  # L face, square index 2
            (12, self.faces["Down"][3]),  # D face, square index 3
            (13, self.faces["Back"][2]),  # B face, square index 2
        ]

        # Fill the one-hot vector
        for i, color in indices:
            one_hot[i * 6 + color] = 1

        return one_hot

    @property
    def up(self):
        return self.faces["Up"]

    @property
    def down(self):
        return self.faces["Down"]

    @property
    def front(self):
        return self.faces["Front"]

    @property
    def back(self):
        return self.faces["Back"]

    @property
    def left(self):
        return self.faces["Left"]

    @property
    def right(self):
        return self.faces["Right"]

    def __eq__(self, other):
        """Simple equality check that works for both Cube and ImmutableCube"""
        return isinstance(other, (Cube, ImmutableCube)) and self.faces == other.faces

    def __hash__(self):
        """Hash implementation matching the reference code"""
        tuples = [tuple([key.lower()] + values) for key, values in self.faces.items()]
        return hash(tuple(tuples))

    def copy(self):
        """Create and return a deep copy of the current cube state."""
        new_cube = Cube()
        new_cube.faces = copy.deepcopy(self.faces)
        return new_cube


class ImmutableCube(Cube):
    """Immutable version of the Cube class"""

    def __init__(self, cube=None):
        """Initialize from another cube or create a fresh cube"""
        super().__init__()  # Create fresh cube
        if cube is not None:
            self.faces = copy.deepcopy(cube.faces)  # Copy state if cube provided

    def change_by(self, move: Move):
        """Returns a new ImmutableCube with the move applied."""
        new_cube = Cube()  # Create regular cube
        new_cube.faces = copy.deepcopy(self.faces)  # Copy current state
        new_cube.change_by(move)  # Use regular cube's change_by
        return ImmutableCube(new_cube)


def get_children_of(cube: Cube):
    """Generate all possible next states from current cube state."""
    imm = ImmutableCube(cube)  # Create ImmutableCube from existing cube
    return (imm.change_by(move) for move in Move)


# def main():
#     pass
# plt.ion()  # Turn on interactive mode once at the start
# cube = Cube(visualize=True)
# cube.plot_3d_cube()

# while True:
#     moves = input("\nEnter move(s): ").strip()

#     if moves == "quit":
#         print("Thanks for playing!")
#         break

#     if moves == "reset":
#         cube = Cube(visualize=True)
#         cube.plot_3d_cube()
#         print("\nCube reset to initial state")
#         continue

#     try:
#         cube.execute_move_sequence(moves)
#         print("\nIs solved:", cube.is_solved())
#     except Exception as e:
#         print(f"Error executing moves: {e}")
#         print("Please try again with valid moves")

# cube = Cube(visualize=True)
# print("\nWelcome to the Interactive 3D Rubik's Cube!")
# print("\nValid moves are:")
# print("L (Left), D (Down), B (Back)")
# print("Add ' for counterclockwise moves (e.g., L', D', B')")
# print("\nYou can input:")
# print("1. A single move (e.g., 'L')")
# print("2. A move with repetition (e.g., 'L4' for four U moves)")
# print("3. Multiple moves separated by commas (e.g., 'L, D2, B4')")
# print("4. Type 'reset' to reset to a fresh cube")
# print("5. Type 'quit' to exit")

# while True:
#     moves = input("\nEnter move(s): ").strip()

#     if moves == "quit":
#         print("Thanks for playing!")
#         break

#     if moves == "reset":
#         plt.close("all")
#         cube = Cube()  # Create fresh cube
#         cube.plot_3d_cube()
#         print("\nCube reset to initial state")
#         continue

#     try:
#         cube.execute_move_sequence(moves)
#         cube.print_raw_arrays()

#         print("\nIs solved:", cube.is_solved())
#     except Exception as e:
#         print(f"Error executing moves: {e}")
#         print("Please try again with valid moves")

# plt.ioff()
# plt.close()


# if __name__ == "__main__":
#     main()
