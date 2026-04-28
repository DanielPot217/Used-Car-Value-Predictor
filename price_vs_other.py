import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator

# 1. Load the dataset
file_path = './data/craigslist_vehicles.csv'
df = pd.read_csv(file_path)

# 2. Filter price outliers so the boxplots aren't squished by multimillion-dollar typos
df_filtered = df[(df['price'] >= 500) & (df['price'] <= 100000)].copy()

# 3. Handle high-cardinality columns (Too many unique values)
# We isolate the top 10 most common manufacturers and models for readability
top_mfg = df_filtered['manufacturer'].value_counts().head(10).index
top_model = df_filtered['model'].value_counts().head(10).index

# Apply the filter: only keep rows if they are in the Top 10 lists
df_plot = df_filtered[
    (df_filtered['manufacturer'].isin(top_mfg)) & 
    (df_filtered['model'].isin(top_model))
]

# 4. Define the categorical features we want to loop through
categorical_features = ['title_status', 'transmission', 'condition', 'manufacturer']


# 5. Set up a grid of subplots (3 rows, 2 columns = 6 slots)
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 16))
axes = axes.flatten() # Flattens the 3x2 matrix into a simple 1D array for easy looping

# 6. Loop through the features and create a box plot for each
for i, feature in enumerate(categorical_features):
    sns.boxplot(
        data=df_plot, 
        x=feature, 
        y='price', 
        ax=axes[i], 
        palette='Set2',
        hue=feature,   # Assigned to hue to avoid palette warnings in modern Seaborn
        legend=False
    )
    
    # Formatting to make it pretty
    axes[i].set_title(f'Price vs. {feature.title()}', fontsize=14, fontweight='bold')
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Price ($)')
    axes[i].tick_params(axis='x', rotation=45) # Tilts the text so it doesn't overlap
    axes[i].yaxis.set_major_locator(MultipleLocator(10000)) # Set y-axis increments to 10,000

# 7. We only have 5 features, but 6 subplot slots. Delete the 6th empty chart.
fig.delaxes(axes[4])
fig.delaxes(axes[5])


# 8. Display the final visualization
plt.tight_layout()
plt.show()