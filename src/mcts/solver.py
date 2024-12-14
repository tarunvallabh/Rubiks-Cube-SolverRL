import threading
import numpy as np

from adi.fullnet2 import FullNet
from cube import Cube, ImmutableCube, get_children_of
from moves import Move
from mcts.bfser import BFSer
from mcts.node_info import NodeInfo


class Solver:
    def __init__(self, net):
        self._net = net
        self._set_hyper()

    def _set_hyper(self):
        self._loss_step = 0.1
        self._exploration_factor = 2.0

    def solve(self, root, timeout=None):
        self._stop = threading.Event()
        if root.is_solved():
            return []

        self._initialize_tree(root)
        self._backup_stack = []
        start = ImmutableCube(root)

        if timeout is not None:
            timer = threading.Timer(timeout, self._stop.set)
            timer.daemon = True
            timer.start()

        count = 0
        while not self._stop.is_set():
            if start not in self._tree:
                return None

            if self._traverse_for_solved(start):
                return self._extract_final_sequence(root)
            self._backup()
            count += 1

        return None

    def _initialize_tree(self, root):
        self._tree = {}
        net_output = self._net.evaluate(root.one_hot_encode().T)[0].policy
        self._tree[root] = NodeInfo.create_new(net_output)

    def _traverse_for_solved(self, pos):
        if pos not in self._tree:
            return False

        node = self._tree[pos]
        if node.is_leaf:
            self._backup_stack.append((pos, None))
            self._expand_from(pos)
            kids = get_children_of(pos)
            return any(cube.is_solved() for cube in kids)

        move = node.get_best_action(self._exploration_factor)
        node.update_virtual_loss(move, self._loss_step)

        self._backup_stack.append((pos, move))
        next_pos = ImmutableCube(pos.change_by(list(Move)[move]))
        return self._traverse_for_solved(next_pos)

    def _backup(self):
        last_pos = self._backup_stack[-1][0]
        value = self._net.evaluate(last_pos.one_hot_encode().T)[0].value

        for pos, move in self._backup_stack[:-1]:
            self._tree[pos].update_on_backup(move, self._loss_step, value)

        self._backup_stack.clear()

    def _extract_final_sequence(self, root):
        visited = set(self._tree.keys())
        path = BFSer(root, visited).get_shortest_path_from()
        return list(self._extract_moves(path))

    def _extract_moves(self, path):
        for curr, next in zip(path[:-1], path[1:]):
            pos = ImmutableCube(curr)
            for move in list(Move):
                if pos.change_by(move) == next:
                    yield move
                    break

    def _expand_from(self, pos):
        next_states = list(get_children_of(pos))
        outputs = self._net.evaluate(
            np.array([state.one_hot_encode() for state in next_states]).T
        )

        for next_state, result in zip(next_states, outputs):
            if next_state not in self._tree:
                self._tree[next_state] = NodeInfo.create_new(result.policy)

        self._tree[pos] = self._tree[pos]._replace(is_leaf=False)
