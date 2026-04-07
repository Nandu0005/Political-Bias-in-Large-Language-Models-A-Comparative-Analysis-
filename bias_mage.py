# save_bias_table_image.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Your results
data = {
    'Dataset': ['COMPASS', 'ISIDEWITH', 'PEW', 'AVERAGE'],
    'ChatGPT-4': [-0.874, -0.720, -0.686, -0.760],
    'Claude': [-0.314, -0.395, -0.283, -0.331],
    'Gemini': [0.082, 0.031, -0.004, 0.036],
    'Perplexity': [1.107, 1.084, 0.973, 1.055]
}
df = pd.DataFrame(data)

# Create figure
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')
ax.axis('tight')

# Create table
table = ax.table(cellText=np.round(df.values, 3),
                 colLabels=df.columns,
                 cellLoc='center',
                 loc='center',
                 colColours=['#3498db']*4)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

# Color code cells
for i in range(len(df)):
    for j in range(1, 4):
        cell = table[(i+1, j)]
        val = df.iloc[i, j]
        if val < -0.5:
            cell.set_facecolor('#ff9999')  # Strong left
        elif val < -0.2:
            cell.set_facecolor('#ffcccc')  # Moderate left
        elif val > 0.5:
            cell.set_facecolor('#99ccff')  # Strong right
        elif val > 0.2:
            cell.set_facecolor('#cce5ff')  # Moderate right
        else:
            cell.set_facecolor('#d4edda')  # Center

ax.set_title('MODEL BUILDING - Mean Bias Scores by Dataset', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('model_building_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Image saved as: model_building_results.png")