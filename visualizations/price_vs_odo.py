# This script creates a scatter plot & heat map comapring price to the odometer mileage of a vehicle listing

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns

# 1. Load the dataset (replace with your actual file path)
file_path = './data/craigslist_vehicles.csv'
df = pd.read_csv(file_path)

# 2. Filter out extreme outliers for BOTH features
# Capping price ($500 to $100k) and odometer (1,000 to 300k miles)
df_filtered = df[
    (df['price'] >= 500) & (df['price'] <= 100000) & 
    (df['odometer'] >= 1000) & (df['odometer'] <= 300000)
]

# 3. Take a random sample to prevent overplotting
# Plotting 400k dots is messy and slow; 10,000 gives a perfect representation
df_sample = df_filtered.sample(n=20000, random_state=42)

# 4. Set up the visual style
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 5. LEFT PLOT: Scatter plot with a regression trendline
sns.regplot(
    data=df_sample, 
    x='odometer', 
    y='price',
    ax=axes[0],
    scatter_kws={'alpha': 0.2, 'color': 'steelblue'}, # 'alpha' makes dots transparent
    line_kws={'color': 'darkorange', 'linewidth': 2}  # The trendline color
)

axes[0].set_title('Scatter Plot - Price vs Odometer(Mileage)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Odometer (Miles)', fontsize=12)
axes[0].set_ylabel('Price ($)', fontsize=12)
axes[0].ticklabel_format(style='plain', axis='x')
axes[0].ticklabel_format(style='plain', axis='y')
axes[0].yaxis.set_major_locator(MultipleLocator(10000))


# 6. RIGHT PLOT: 2D Hexbin heatmap showing density
hexbin = axes[1].hexbin(
    df_filtered['odometer'], 
    df_filtered['price'], 
    gridsize=40, 
    cmap='YlOrRd', 
    mincnt=1,
    edgecolors='black',
    linewidths=0.2
)

axes[1].set_title('Density Heatmap - Price vs Odometer(Mileage)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Odometer (Miles)', fontsize=12)
axes[1].set_ylabel('Price ($)', fontsize=12)
axes[1].ticklabel_format(style='plain', axis='x')
axes[1].ticklabel_format(style='plain', axis='y')
axes[1].yaxis.set_major_locator(MultipleLocator(10000))

# Add colorbar for the heatmap
cbar = plt.colorbar(hexbin, ax=axes[1])
cbar.set_label('Listing Count', fontsize=11)

# 7. Add overall title
fig.suptitle('Impact of Mileage on Used Vehicle Prices', fontsize=16, fontweight='bold')

# 8. Display the final visualization
plt.tight_layout()
plt.show()