A Python3 library for running a [Monte Carlo tree search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search), either traditionally by drilling down to end game states or with expert policies as might be provided by a neural network.

Adapted from **Version:** 1.3.1 of [ImparaAI/monte-carlo-tree-search](https://github.com/ImparaAI/monte-carlo-tree-search).

# Monte Carlo tree search basics

The Monte Carlo tree search (MCTS) algorithm can help with making a decision from a number of options. It avoids exploring every possible option by randomly sampling a small number of pathways and picking the move with the highest probability of victory. This is commonly applied to games like chess or go where it's useful to know what move should come next if you want to win the game.

MCTS works by expanding the search tree to figure out which moves (or child/subsequent states) are likely to produce a positive result if chosen. While time is available, the algorithm continues to explore the tree, always slightly favoring the direction that has either proven to be fruitful or is less explored. When no time is left, the most explored direction is chosen.

The search tree expansion can be done in two different ways:

- **Traditional**: At least one random rollout to a game's end state (e.g. win, loss, tie) for each move under evaluation so the algorithm can make a choice.
- **Expert policy (i.e. neural network)**: Instead of expensively rolling all the way out to a game's end state ask an expert (a neural network for example) which move is most likely to produce a positive outcome.

For a deeper dive into the topic, check out [this article](http://tim.hibal.org/blog/alpha-zero-how-and-why-it-works/).

# This library

As the user of this library, you only have to provide:

- A function that finds the direct children of each search tree node (called the **`child_finder`**)
- A function for evaluating nodes for end state outcomes (called the **`node_evaluator`**)
-- *(Not necessary with neural network)*

# Usage

Create a new Monte Carlo tree:

```python
from chess import Game
from montecarlo.node import Node
from montecarlo.montecarlo import MonteCarlo

chess_game = Game()
montecarlo = MonteCarlo(Node(chess_game))
```

The root node describes your current game state. This state will be used by you later in the **`child_finder`** and the **`node_evaluator`**.

For the sake of demonstration, we will assume you have a generic `Game` library that can tell you what moves are possible and allows you to perform those moves to change the game's state.

## Traditional Monte Carlo

Add a **`child_finder`** and a **`node_evaluator`**:

```python
def child_finder(node, montecarlo):
	for move in node.state.get_possible_moves():
		child = Node(deepcopy(node.state)) #or however you want to construct the child's state
		child.state.move(move) #or however your library works
		node.add_child(child)

def node_evaluator(node, montecarlo):
	if node.state.won():
		return 1
	elif node.state.lost():
		return -1

montecarlo.child_finder = child_finder
montecarlo.node_evaluator = node_evaluator
```

The **`child_finder`** should add any child nodes to the parent node passed into the function, if there are any. If there are none, the parent should be in an end state, so the **`node_evaluator`** should return a value between `-1` and `1`.

## Expert policy (AI)

If you have an expert policy that you can apply to the children as they're being generated, the library will recognize that it doesn't need to make the costly drill down to an end state. If your neural net produces both an expert policy value for the children and a win value for the parent node, you can skip declaring the `node_evaluator` altogether.

```python
def child_finder(node, montecarlo):
	win_value, expert_policy_values = neural_network.predict(node.state)

	for move in node.state.get_possible_moves():
		child = Node(deepcopy(node.state))
		child.state.move(move)
		child.player_number = child.state.whose_turn()
		child.policy_value = get_child_policy_value(child, expert_policy_values) #should return a probability value between 0 and 1
		node.add_child(child)

	node.update_win_value(win_value)

montecarlo.child_finder = child_finder
```

## Simulate and make a choice

Run the simulations:

```python
montecarlo.simulate(50) #number of expansions to run. higher is typically more accurate at the cost of processing time
```

Once the simulations have run you can ask the instance to make a choice:

```python
chosen_child_node = montecarlo.make_choice()
chosen_child_node.state.do_something()
```

After you've chosen a new root node, you can override it on the `montecarlo` instance and do more simulations from the new position in the tree.

```python
montecarlo.root_node = montecarlo.make_choice()
```

If you're training a neural network, you may want to make a more exploratory choice for the first N moves of a game:

```python
montecarlo.root_node = montecarlo.make_exploratory_choice()
```

This won't provide a purely random choice, rather it will be random with a bias favoring the more explored pathways.

## Turn-based environments

If you are modeling a turn-based environment (e.g. a two player board game), set the `player_number` on each node so the selection process can invert child win values:

```python
node = Node(state)
node.player_number = 1
```

It doesn't matter what this number is (you can use 1 and 2 or 5 and 6), only that it is consistent with other nodes.

## Tweaking the discovery factor

When building a new child node, you can change the rate at which discovery is preferred:

```python
node = Node(state)
node.discovery_factor = 0.2 #0.35 by default, can be between 0 and 1
```

The closer this number is to 1, the more discovery will be favored over demonstrated value in later simulations.
