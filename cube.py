import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
import numpy as np
import re


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
            "Left": [self.cube_colors["red"] for _ in range(4)],
            "Right": [self.cube_colors["orange"] for _ in range(4)],
        }

        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111, projection="3d")
        plt.ion()  # Turn on interactive mode

    def plot_3d_cube(self):
        """Plot the cube in 3D using our previous visualization code"""
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
                    "back": self.faces["Back"][0],
                    "left": self.faces["Left"][0],
                    "top": self.faces["Up"][0],
                },
            },
            {
                "pos": (1, 1, 1),
                "colors": {
                    "back": self.faces["Back"][1],
                    "right": self.faces["Right"][1],
                    "top": self.faces["Up"][1],
                },
            },
            # Back face, bottom row
            {
                "pos": (0, 1, 0),
                "colors": {
                    "back": self.faces["Back"][2],
                    "left": self.faces["Left"][2],
                    "bottom": self.faces["Down"][2],
                },
            },
            {
                "pos": (1, 1, 0),
                "colors": {
                    "back": self.faces["Back"][3],
                    "right": self.faces["Right"][3],
                    "bottom": self.faces["Down"][3],
                },
            },
        ]

        # Plot all cubes
        for config in cube_configs:
            plot_cube(self.ax, config["pos"], config["colors"])

        self.ax.set_box_aspect([2, 2, 2])
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_xlim(0, 2)
        self.ax.set_ylim(0, 2)
        self.ax.set_zlim(0, 2)
        self.ax.view_init(elev=20, azim=-60)

        plt.title("2x2x2 Rubik's Cube")
        plt.draw()
        plt.pause(0.5)

    # [All the existing move methods (move_left, move_right, etc.) remain the same]
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
        self.faces["Front"][0], self.faces["Front"][2] = up_edge[0], up_edge[1]
        self.faces["Down"][0], self.faces["Down"][2] = front_edge[0], front_edge[1]
        self.faces["Back"][0], self.faces["Back"][2] = down_edge[0], down_edge[1]
        self.faces["Up"][0], self.faces["Up"][2] = back_edge[0], back_edge[1]

        self.plot_3d_cube()  # Use our 3D visualization instead

    def move_right(self):
        self.rotate_face_clockwise(self.faces["Right"])

        # store the edge pieces
        up_edge = [self.faces["Up"][1], self.faces["Up"][3]]
        front_edge = [self.faces["Front"][1], self.faces["Front"][3]]
        down_edge = [self.faces["Down"][1], self.faces["Down"][3]]
        back_edge = [self.faces["Back"][1], self.faces["Back"][3]]

        # update the edge pieces
        # up adjacent to back
        self.faces["Up"][1], self.faces["Up"][3] = front_edge[0], front_edge[1]
        # up -> front
        self.faces["Front"][1], self.faces["Front"][3] = down_edge[0], down_edge[1]
        # front -> down
        self.faces["Down"][1], self.faces["Down"][3] = back_edge[0], back_edge[1]
        # down -> back
        self.faces["Back"][1], self.faces["Back"][3] = up_edge[0], up_edge[1]

        self.plot_3d_cube()

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
        self.faces["Left"][0], self.faces["Left"][1] = front_edge[0], front_edge[1]
        # Right's top edge goes to Back
        self.faces["Back"][0], self.faces["Back"][1] = left_edge[0], left_edge[1]
        # Back's top edge goes to Left
        self.faces["Right"][0], self.faces["Right"][1] = back_edge[0], back_edge[1]
        # Left's top edge goes to Front
        self.faces["Front"][0], self.faces["Front"][1] = right_edge[0], right_edge[1]

        self.plot_3d_cube()

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

        self.plot_3d_cube()

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

        self.plot_3d_cube()

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

        self.plot_3d_cube()

    def move_left_prime(self):
        """Counterclockwise Left face rotation"""
        self.rotate_face_counterclockwise(self.faces["Left"])

        # Store the edge pieces
        up_edge = [self.faces["Up"][0], self.faces["Up"][2]]
        front_edge = [self.faces["Front"][0], self.faces["Front"][2]]
        down_edge = [self.faces["Down"][0], self.faces["Down"][2]]
        back_edge = [self.faces["Back"][0], self.faces["Back"][2]]

        # Update the edge pieces (reverse of clockwise)
        self.faces["Up"][0], self.faces["Up"][2] = front_edge[0], front_edge[1]
        self.faces["Front"][0], self.faces["Front"][2] = down_edge[0], down_edge[1]
        self.faces["Down"][0], self.faces["Down"][2] = back_edge[0], back_edge[1]
        self.faces["Back"][0], self.faces["Back"][2] = up_edge[0], up_edge[1]

        self.plot_3d_cube()

    def move_right_prime(self):
        """Counterclockwise Right face rotation"""
        self.rotate_face_counterclockwise(self.faces["Right"])

        up_edge = [self.faces["Up"][1], self.faces["Up"][3]]
        front_edge = [self.faces["Front"][1], self.faces["Front"][3]]
        down_edge = [self.faces["Down"][1], self.faces["Down"][3]]
        back_edge = [self.faces["Back"][1], self.faces["Back"][3]]

        self.faces["Up"][1], self.faces["Up"][3] = back_edge[0], back_edge[1]
        self.faces["Back"][1], self.faces["Back"][3] = down_edge[0], down_edge[1]
        self.faces["Down"][1], self.faces["Down"][3] = front_edge[0], front_edge[1]
        self.faces["Front"][1], self.faces["Front"][3] = up_edge[0], up_edge[1]

        self.plot_3d_cube()

    def move_up_prime(self):
        """Counterclockwise Up face rotation"""
        self.rotate_face_counterclockwise(self.faces["Up"])

        front_edge = [self.faces["Front"][0], self.faces["Front"][1]]
        right_edge = [self.faces["Right"][0], self.faces["Right"][1]]
        back_edge = [self.faces["Back"][0], self.faces["Back"][1]]
        left_edge = [self.faces["Left"][0], self.faces["Left"][1]]

        self.faces["Front"][0], self.faces["Front"][1] = left_edge[0], left_edge[1]
        self.faces["Right"][0], self.faces["Right"][1] = front_edge[0], front_edge[1]
        self.faces["Back"][0], self.faces["Back"][1] = right_edge[0], right_edge[1]
        self.faces["Left"][0], self.faces["Left"][1] = back_edge[0], back_edge[1]

        self.plot_3d_cube()

    def move_down_prime(self):
        """Counterclockwise Down face rotation"""
        self.rotate_face_counterclockwise(self.faces["Down"])

        front_edge = [self.faces["Front"][2], self.faces["Front"][3]]
        right_edge = [self.faces["Right"][2], self.faces["Right"][3]]
        back_edge = [self.faces["Back"][2], self.faces["Back"][3]]
        left_edge = [self.faces["Left"][2], self.faces["Left"][3]]

        self.faces["Front"][2], self.faces["Front"][3] = right_edge[0], right_edge[1]
        self.faces["Right"][2], self.faces["Right"][3] = back_edge[0], back_edge[1]
        self.faces["Back"][2], self.faces["Back"][3] = left_edge[0], left_edge[1]
        self.faces["Left"][2], self.faces["Left"][3] = front_edge[0], front_edge[1]

        self.plot_3d_cube()

    def move_front_prime(self):
        """Counterclockwise Front face rotation"""
        self.rotate_face_counterclockwise(self.faces["Front"])

        up_edge = [self.faces["Up"][2], self.faces["Up"][3]]
        right_edge = [self.faces["Right"][0], self.faces["Right"][2]]
        down_edge = [self.faces["Down"][0], self.faces["Down"][1]]
        left_edge = [self.faces["Left"][1], self.faces["Left"][3]]

        self.faces["Up"][2], self.faces["Up"][3] = right_edge[0], right_edge[1]
        self.faces["Right"][0], self.faces["Right"][2] = down_edge[0], down_edge[1]
        self.faces["Down"][0], self.faces["Down"][1] = left_edge[0], left_edge[1]
        self.faces["Left"][1], self.faces["Left"][3] = up_edge[0], up_edge[1]

        self.plot_3d_cube()

    def move_back_prime(self):
        """Counterclockwise Back face rotation"""
        self.rotate_face_counterclockwise(self.faces["Back"])

        up_edge = [self.faces["Up"][0], self.faces["Up"][1]]
        left_edge = [self.faces["Left"][0], self.faces["Left"][2]]
        down_edge = [self.faces["Down"][2], self.faces["Down"][3]]
        right_edge = [self.faces["Right"][1], self.faces["Right"][3]]

        self.faces["Up"][0], self.faces["Up"][1] = left_edge[0], left_edge[1]
        self.faces["Left"][0], self.faces["Left"][2] = down_edge[0], down_edge[1]
        self.faces["Down"][2], self.faces["Down"][3] = right_edge[0], right_edge[1]
        self.faces["Right"][1], self.faces["Right"][3] = up_edge[0], up_edge[1]

        self.plot_3d_cube()

    # [Add all other move methods here with same structure]

    def execute_move(self, move):
        """Execute a single move on the cube."""
        match = re.match(r"([FBLRUD])('?)(\d*)", move.strip().upper())
        if not match:
            print(f"Invalid move: {move}")
            return

        face, prime, repetition = match.groups()
        repetition = int(repetition) if repetition else 1

        for _ in range(repetition):
            if prime:  # Counterclockwise moves
                if face == "F":
                    self.move_front_prime()
                elif face == "B":
                    self.move_back_prime()
                elif face == "L":
                    self.move_left_prime()
                elif face == "R":
                    self.move_right_prime()
                elif face == "U":
                    self.move_up_prime()
                elif face == "D":
                    self.move_down_prime()
            else:  # Clockwise moves
                if face == "F":
                    self.move_front()
                elif face == "B":
                    self.move_back()
                elif face == "L":
                    self.move_left()
                elif face == "R":
                    self.move_right()
                elif face == "U":
                    self.move_up()
                elif face == "D":
                    self.move_down()

    def execute_move_sequence(self, moves):
        """Execute a sequence of moves on the cube."""
        if "," in moves:
            move_list = moves.split(",")
            for move in move_list:
                self.execute_move(move)
        else:
            self.execute_move(moves)

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


def main():
    cube = Cube()
    print("\nWelcome to the Interactive 3D Rubik's Cube!")
    print("\nValid moves are:")
    print("F (Front), B (Back), L (Left), R (Right), U (Up), D (Down)")
    print("\nYou can input:")
    print("1. A single move (e.g., 'F')")
    print("2. A move with repetition (e.g., 'U4' for four U moves)")
    print("3. Multiple moves separated by commas (e.g., 'F, R2, U4')")
    print("4. Type 'quit' to exit")

    # Initial plot
    cube.plot_3d_cube()

    while True:
        moves = input("\nEnter move(s): ").strip()

        if moves.lower() == "quit":
            print("Thanks for playing!")
            break

        try:
            cube.execute_move_sequence(moves)
            # cube.print_raw_arrays()

            print("\nIs solved:", cube.is_solved())
        except Exception as e:
            print(f"Error executing moves: {e}")
            print("Please try again with valid moves")

    plt.ioff()
    plt.close()


if __name__ == "__main__":
    main()
