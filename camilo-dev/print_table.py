from tabulate import tabulate

data = {'Random Bot': {'Random Bot': None, 'Max Bot': 0.04, 'Heuristics Bot': 0.0, 'BL DQN Bot': 0.14, 'Our DQN Bot': 0.26}, 'Max Bot': {'Random Bot': 0.96, 'Max Bot': None, 'Heuristics Bot': 0.22, 'BL DQN Bot': 0.28, 'Our DQN Bot': 0.82}, 'Heuristics Bot': {'Random Bot': 1.0, 'Max Bot': 0.78, 'Heuristics Bot': None, 'BL DQN Bot': 0.82, 'Our DQN Bot': 0.96}, 'BL DQN Bot': {'Random Bot': 0.86, 'Max Bot': 0.72, 'Heuristics Bot': 0.18, 'BL DQN Bot': None, 'Our DQN Bot': 0.84}, 'Our DQN Bot': {'Random Bot': 0.74, 'Max Bot': 0.18, 'Heuristics Bot': 0.04, 'BL DQN Bot': 0.16, 'Our DQN Bot': None}}

headers = list(data.keys())
rows = [[data[row].get(col) for col in headers] for row in headers]
print(tabulate(rows, headers=headers, showindex=headers, tablefmt="grid"))