import random
import time


class MonteCarlo:
    def __init__(self, root_node, mins_timeout=None):
        self.root_node = root_node
        self.solution = None
        self.child_finder = None
        self.node_evaluator = lambda child, montecarlo: None
        self.stats_expansion_count = 0
        self.stats_failed_expansion_count = 0
        self.mins_timeout = mins_timeout

    def make_choice(self):
        best_children = []
        most_visits = float("-inf")

        for child in self.root_node.children:
            if child.visits > most_visits:
                most_visits = child.visits
                best_children = [child]
            elif child.visits == most_visits:
                best_children.append(child)

        return random.choice(best_children)

    def make_exploratory_choice(self):
        children_visits = map(lambda child: child.visits, self.root_node.children)
        children_visit_probabilities = [
            visit / self.root_node.visits for visit in children_visits
        ]
        random_probability = random.uniform(0, 1)
        probabilities_already_counted = 0.0

        for i, probability in enumerate(children_visit_probabilities):
            if probabilities_already_counted + probability >= random_probability:
                return self.root_node.children[i]

            probabilities_already_counted += probability

    def simulate(self, expansion_count=1):
        i = 0
        
        start_time = time.time()

        while expansion_count is None or i < expansion_count:
            i += 1

            if self.solution is not None:
                return

            if self.mins_timeout is not None:
                curr_time = time.time()
                duration = curr_time - start_time

                if duration > (self.mins_timeout * 60):
                    print("reached timelimit, stopping expansion on current node")
                    return

            current_node = self.root_node

            while current_node.expanded:
                current_node = current_node.get_preferred_child(self.root_node)

            self.expand(current_node)

    def expand(self, node):
        self.stats_expansion_count += 1
        self.child_finder(node, self)

        for child in node.children:
            child_win_value = self.node_evaluator(child, self)

            if child_win_value != None:
                child.update_win_value(child_win_value)

            if not child.is_scorable():
                self.random_rollout(child)
                child.children = []

        if len(node.children):
            node.expanded = True
        else:
            self.stats_failed_expansion_count += 1

    def random_rollout(self, node):
        self.child_finder(node, self)
        child = random.choice(node.children)
        node.children = []
        node.add_child(child)
        child_win_value = self.node_evaluator(child, self)

        if child_win_value != None:
            node.update_win_value(child_win_value)
        else:
            self.random_rollout(child)

    def print_tree(self, f):
        f.write("graph\n{\n")
        self.root_node.print_node(f, 0, self.root_node, "a")
        f.write("}\n")
