from tabulate import tabulate

data = {'Random Bot': {'Random Bot': None, 'P1 DQN Bot': 0.2, 'P2 DQN Bot': 0.6, 'P3 DQN Bot': 0.4}, 'P1 DQN Bot': {'Random Bot': 0.8, 'P1 DQN Bot': None, 'P2 DQN Bot': 0.5, 'P3 DQN Bot': 0.8}, 'P2 DQN Bot': {'Random Bot': 0.4, 'P1 DQN Bot': 0.5, 'P2 DQN Bot': None, 'P3 DQN Bot': 0.5}, 'P3 DQN Bot': {'Random Bot': 0.6, 'P1 DQN Bot': 0.2, 'P2 DQN Bot': 0.5, 'P3 DQN Bot': None}}
headers = list(data.keys())
rows = [[data[row].get(col) for col in headers] for row in headers]
print(tabulate(rows, headers=headers, showindex=headers, tablefmt="grid"))