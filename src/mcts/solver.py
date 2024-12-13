import threading
from typing import List, Iterable, Union
import numpy as np

from adi.fullnet import FullNet
from cube import Cube, ImmutableCube, get_children_of
from moves import Move
from mcts.bfser import BFSer
from mcts.node_info import NodeInfo


class Solver:
    def __init__(self, net: FullNet):
        self._net = net
        self._set_hyper()

    def _set_hyper(self):
        self._loss_step = 0.1
        self._exploration_factor = 2.0

    def solve(
        self, root: Cube, timeout: Union[int, None] = None
    ) -> Union[List[Move], None]:
        # print("\nStarting solve method...")
        # print("Input cube is_solved():", root.is_solved())

        self._stop = threading.Event()
        if root.is_solved():
            print("Cube is already solved, returning empty move list")
            return []

        # print("\nInitializing tree...")
        self._initialize_tree(root)
        self._backup_stack = []

        # Debug hash values
        # print("Original cube hash:", hash(root))
        # print("Tree keys:", [hash(k) for k in self._tree.keys()])

        root = ImmutableCube(root)
        # print("ImmutableCube hash:", hash(root))
        # print("Is original in tree?", root in self._tree)
        # print("Tree size:", len(self._tree))

        if timeout is not None:
            timer = threading.Timer(timeout, lambda: self._stop.set())
            timer.daemon = True
            timer.start()

        count = 0
        while not self._stop.is_set():
            # print(f"\nTraversal {count}...")
            if root not in self._tree:
                print("ERROR: Root not in tree!")
                print("Current tree keys:", [hash(k) for k in self._tree.keys()])
                return None

            if self._traverse_for_solved(root):
                # print(f"Solution found in {count} iterations!")
                return self._extract_final_sequence(root)
            self._backup()
            count += 1

        return None

    def _initialize_tree(self, root: Cube):
        self._tree = dict()
        policy = self._net.evaluate(root.one_hot_encode().T)[0].policy
        self._tree[root] = NodeInfo.create_new(policy)

    def _traverse_for_solved(self, current: ImmutableCube) -> bool:
        assert current in self._tree
        if self._tree[current].is_leaf:
            self._backup_stack.append((current, None))
            self._expand_from(current)
            return any((child.is_solved() for child in get_children_of(current)))
        best_move = self._tree[current].get_best_action(self._exploration_factor)
        self._tree[current].update_virtual_loss(best_move, self._loss_step)

        self._backup_stack.append((current, best_move))
        return self._traverse_for_solved(
            ImmutableCube(current.change_by(list(Move)[best_move]))
        )

    def _backup(self):
        last_state = self._backup_stack[-1][0]
        propagation_value = self._net.evaluate(last_state.one_hot_encode().T)[0].value
        for state, move in self._backup_stack[:-1]:
            self._tree[state].update_on_backup(move, self._loss_step, propagation_value)
        self._backup_stack.clear()

    def _extract_final_sequence(self, root: Cube) -> List[Move]:
        path = BFSer(root, set(self._tree.keys())).get_shortest_path_from()
        return list(self._extract_moves(path))

    def _extract_moves(self, path: List[Cube]) -> Iterable[Move]:
        for cur, next in zip(path[:-1], path[1:]):
            imm = ImmutableCube(cur)
            for move in list(Move):
                if imm.change_by(move) == next:
                    yield move
                    break

    def _expand_from(self, current: Cube):
        children = list(get_children_of(current))
        children_evals = self._net.evaluate(
            np.array([child.one_hot_encode() for child in children]).T
        )
        children_policies = [eval.policy for eval in children_evals]
        for child, policy in zip(children, children_policies):
            if child not in self._tree:
                self._tree[child] = NodeInfo.create_new(policy)
        self._tree[current] = self._tree[current]._replace(is_leaf=False)
