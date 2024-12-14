from collections import deque
from typing import List, Set
from cube import Cube, get_children_of


class BFSer:
    def __init__(self, root: Cube, nodes: Set[Cube]):
        self._nodes = nodes
        self._root = root
        assert Cube() in nodes, "Solved cube must be in the set of nodes"

    def get_shortest_path_from(self):
        self._create_edges()
        self._init_auxiliary()
        self._perform_bfs()
        return self._extract_path()

    def _create_edges(self):
        self._edges = {}
        for current in self._nodes:
            possible_moves = get_children_of(current)
            self._edges[current] = [
                move for move in possible_moves if move in self._nodes
            ]

    def _init_auxiliary(self):
        self._visited = set()
        self._parent = {}
        self._parent[self._root] = None

    def _perform_bfs(self):
        search_queue = deque([self._root])
        self._visited.add(self._root)

        while search_queue:
            current_state = search_queue.popleft()
            for next_move in self._edges[current_state]:
                if next_move not in self._visited:
                    self._visited.add(next_move)
                    self._parent[next_move] = current_state

                    if next_move.is_solved():
                        return
                    search_queue.append(next_move)

    def _extract_path(self):
        path_states = []
        current = Cube()

        while True:
            path_states.append(current)
            if self._parent[current] is None:
                break
            current = self._parent[current]

        return list(reversed(path_states))
