import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from matplotlib.ticker import ScalarFormatter

df = pd.read_csv('healthcare_cleaned.csv')


df.hist(figsize=(15, 20), bins=10, density=True)
plt.suptitle("Histograms")
plt.show()

plt.figure(figsize=(12, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


df_filtered = df.copy()
df_filtered['dvisits'] = df_filtered['dvisits'].astype(int)

all_features = [
    'sex', 'age', 'income', 'levyplus', 'freepor', 'freerepa', 
    'illness', 'actdays', 'hscore', 'chcond1', 'chcond2', 
    'nondocco', 'hospadmi', 'hospdays', 'prescrib', 'nonprescr', 
    'any_insurance', 'health_status_severe', 'recent_hospital', 
    'medication_user', 'any_chronic'
]

count_vars = ['actdays', 'nondocco', 'hospadmi', 'hospdays', 'hscore', 'illness', 'prescrib', 'nonprescr']

def apply_smart_binning(data, column):

    if column in count_vars:
        return pd.cut(data[column], 
                      bins=[-np.inf, 0, 1, 2, np.inf], 
                      labels=['0', '1', '2', '3+']).astype(str)
    elif data[column].nunique() > 10:
        return pd.qcut(data[column], q=5, duplicates='drop').astype(str)
    else:
        return data[column].astype(str)

for i in range(0, len(all_features), 4):
    features = all_features[i : i+4]
    n_cols = 2 
    n_rows = math.ceil(len(features) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
    axes = axes.flatten()

    for idx, col in enumerate(features):
        temp_df = df_filtered.copy()
        temp_df[col] = apply_smart_binning(temp_df, col)

        counts = (temp_df.groupby([col, 'dvisits'])
                .size()
                .unstack(fill_value=0)
                .stack()
                .reset_index(name='count'))
        
        total_per_class = counts.groupby(col)['count'].transform('sum')
        counts['percentage'] = (counts['count'] / total_per_class) * 100

        order = ['0', '1', '2', '3+'] if col in count_vars else sorted(temp_df[col].unique())

        sns.barplot(
            data=counts, 
            x=col, 
            y='percentage', 
            hue='dvisits', 
            ax=axes[idx], 
            palette='viridis',
            order=order
        )
        
        axes[idx].set_yscale('log')
        axes[idx].set_ylim(0.1, 100) 
        axes[idx].yaxis.set_major_formatter(ScalarFormatter())
        axes[idx].set_title(f'LOG Distribution: {col}', fontsize=14, weight='bold')
        axes[idx].set_ylabel('Percentage (Log %)')
        
        if idx == 0:
            axes[idx].legend(title='dvisits', bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            axes[idx].get_legend().remove()

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

for i in range(0, len(all_features), 4):
    features = all_features[i : i+4]
    n_cols = 2 
    n_rows = math.ceil(len(features) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
    axes = axes.flatten()

    for idx, col in enumerate(features):
        temp_df = df_filtered.copy()
        temp_df[col] = apply_smart_binning(temp_df, col)

        order = ['0', '1', '2', '3+'] if col in count_vars else sorted(temp_df[col].unique())

        sns.barplot(
            data=temp_df, 
            x=col, 
            y='dvisits', 
            ax=axes[idx], 
            palette='magma',
            errorbar=('ci', 95),
            order=order
        )
        
        axes[idx].set_title(f'Mean dvisits vs {col}', fontsize=14, weight='bold')
        axes[idx].set_ylabel('dvisits (Mean)')
        axes[idx].tick_params(axis='x', rotation=30)
        
        group_mean = temp_df.groupby(col)['dvisits'].mean().sort_values(ascending=False)

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()