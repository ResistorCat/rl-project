import pandas as pd
import matplotlib.pyplot as plt

# El número que se usa para calcular el WIN RATIO
WINDOW = 100

df = pd.read_csv("results/our_dqn.monitor_parcial.csv", comment='#')

df['win'] = (df['r'] > 0).astype(int)

df[f'win_ratio_{WINDOW}'] = df['win'].rolling(window=WINDOW).mean()

plt.figure()
plt.plot(df.index + 1, df[f'win_ratio_{WINDOW}'], label=f'Win ratio (últimos {WINDOW})')
plt.xlabel('Episodio')
plt.ylabel(f'Ratio de victorias (últimos {WINDOW})')
plt.title('Evolución del ratio de victorias')
plt.legend()
plt.ylim(0.5, 1)
plt.show()
