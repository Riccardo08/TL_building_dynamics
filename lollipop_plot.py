import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TODO: fare un pandas dataframe formato da caso (variabile categorica), value1(metrica di benchmark), valore2(metrica del caso)
# Create a dataframe
# value1 = np.random.uniform(size=20)
# value2 = value1 + np.random.uniform(size=20) / 4
# df = pd.DataFrame({'group': list(map(int, range(1, 10))), 'value1': value1, 'value2': value2})

df = pd.read_csv('month_cases.csv', encoding='latin1', sep=';')

# Reorder it following the values of the first value:
ordered_df = df.sort_values(by='cases')
my_range = range(1, len(df.index) + 1)

# The horizontal plot is made using the hline function
# TODO: definire i benchmark per ogni caso
plt.hlines(y=my_range, xmin=ordered_df['value1'], xmax=ordered_df['value2'], color='grey', alpha=0.4)
plt.scatter(ordered_df['value1'], my_range, color='skyblue', alpha=1, label='value1')
plt.scatter(ordered_df['value2'], my_range, color='green', alpha=0.4, label='value2')
plt.legend()

# Add title and axis names
plt.yticks(my_range, ordered_df['group'])
plt.title("Comparison of the value 1 and the value 2", loc='left')
plt.xlabel('Value of the variables')
plt.ylabel('Group')

# Show the graph
plt.show()