from enum import IntEnum


class Move(IntEnum):
    LEFT = 1
    LEFT_PRIME = 2  # Separate entry for prime move
    DOWN = 3
    DOWN_PRIME = 4
    BACK = 5
    BACK_PRIME = 6
