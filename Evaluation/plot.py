import pandas as pd
import matplotlib.pyplot as plt
import json
import seaborn as sns
import numpy as np

# Use the poster style
plt.style.use('seaborn-v0_8-poster')

with open('skyview_eval_scores.json', 'r') as file:
    data = json.load(file)

df = pd.DataFrame(data)

means = { 
    'Model': ['LLaVA', 'GPT'],
    'Average Score': [df['llava'].mean(), df['gpt'].mean()]
}
mean_df = pd.DataFrame(means)

# Create the bar chart
plt.figure(figsize=(8, 6))

# Make the plot
bars = plt.bar(mean_df['Model'], mean_df['Average Score'], width=0.7)

# Customize it for presentation
plt.title('SkyView Model Performance Comparison', fontsize=22, pad=15)
plt.ylabel('Average Score (%)', fontsize=18)
plt.xlabel('', fontsize=18)  

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.ylim(bottom=0, top=100)

# Add labels
plt.text(0, mean_df["Average Score"][0] + 2, f'{mean_df["Average Score"][0]:.1f}%', ha='center', fontsize=16, fontweight='bold')
plt.text(0.9, mean_df["Average Score"][1] + 3, f'{mean_df["Average Score"][1]:.1f}%', fontsize=16, fontweight='bold')
plt.tight_layout()

plt.savefig('paper_poster_comparison.png', dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.show()
