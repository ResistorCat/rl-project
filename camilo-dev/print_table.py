from tabulate import tabulate

data = {'Random Bot': {'Random Bot': None, 'Max Bot': 0.06, 'Heuristics Bot': 0.0, 'DQN Bot': 0.1}, 'Max Bot': {'Random Bot': 0.94, 'Max Bot': None, 'Heuristics Bot': 0.24, 'DQN Bot': 0.32}, 'Heuristics Bot': {'Random Bot': 1.0, 'Max Bot': 0.76, 'Heuristics Bot': None, 'DQN Bot': 0.92}, 'DQN Bot': {'Random Bot': 0.9, 'Max Bot': 0.68, 'Heuristics Bot': 0.08, 'DQN Bot': None}}

headers = list(data.keys())
rows = [[data[row].get(col) for col in headers] for row in headers]
print(tabulate(rows, headers=headers, showindex=headers, tablefmt="grid"))