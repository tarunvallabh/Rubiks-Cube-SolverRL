from collections import deque
from typing import List, Set
from cube import Cube, get_children_of


class BFSer:
    def __init__(self, root: Cube, nodes: Set[Cube]):
        self._nodes = nodes
        self._root = root
        assert Cube() in nodes, "Solved cube must be in the set of nodes"

    def get_shortest_path_from(self) -> List[Cube]:
        self._create_edges()
        self._init_auxiliary()
        self._perform_bfs()
        return self._extract_path()

    def _create_edges(self):
        self._edges = dict()
        for node in self._nodes:
            self._edges[node] = [
                child for child in get_children_of(node) if child in self._nodes
            ]

    def _init_auxiliary(self):
        self._visited = set()
        self._parent = dict()
        self._parent[self._root] = None

    def _perform_bfs(self):
        todo = deque()
        todo.append(self._root)
        self._visited.add(self._root)

        while todo:
            current = todo.popleft()
            for child in self._edges[current]:
                if child not in self._visited:
                    self._visited.add(child)
                    self._parent[child] = current

                    if child.is_solved():
                        return
                    todo.append(child)

    def _extract_path(self):
        current = Cube()  # Start from solved cube
        path = [current]
        while self._parent[current] is not None:
            current = self._parent[current]
            path.append(current)
        return list(reversed(path))
