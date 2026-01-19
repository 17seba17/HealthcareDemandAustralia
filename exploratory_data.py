import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from matplotlib.ticker import ScalarFormatter


df = pd.read_csv('healthcare_cleaned.csv')
df.info()
df.head()
df.hist(figsize=(15, 20), bins=10, density=True)
plt.suptitle("Histograms")
plt.show()


corr_matrix = df.corr()

sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
plt.title("Correlation matrix")
plt.show()

print("We see zero-inflation patterns!")

correlations = df.corr()['dvisits'].sort_values(ascending=False)
print(correlations)




df_filtered = df[df['dvisits'] >= 0].copy()
df_filtered['dvisits'] = df_filtered['dvisits'].astype(int)

all_features = [
    'sex', 'age', 'income', 'levyplus'
    , 'freepor', 'freerepa','illness', 'actdays',
     'hscore', 'chcond1', 'chcond2', 'nondocco', 
    'hospadmi', 'hospdays', 'prescrib', 'nonprescr', 
    'any_insurance', 'health_status_severe', 'recent_hospital', 'medication_user', 
    'any_chronic'
]

for i in range(0, len(all_features), 4):
    features = all_features[i : i+4]

    n_cols = 2 
    n_rows = math.ceil(len(features) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(features):
        temp_df = df_filtered.copy()
        
        if temp_df[col].nunique() > 10:
            temp_df[col] = pd.qcut(temp_df[col], q=5, duplicates='drop').astype(str)
        else:
            temp_df[col] = temp_df[col].astype(str)

        counts = (temp_df.groupby([col, 'dvisits'])
                .size()
                .unstack(fill_value=0)
                .stack()
                .reset_index(name='count'))
        
        total_per_class = counts.groupby(col)['count'].transform('sum')
        counts['percentage'] = (counts['count'] / total_per_class) * 100

        sns.barplot(
            data=counts, 
            x=col, 
            y='percentage', 
            hue='dvisits', 
            ax=axes[i], 
            palette='viridis'
        )
        
        axes[i].set_yscale('log')
        axes[i].set_ylim(0.1, 100) 
        
        axes[i].yaxis.set_major_formatter(ScalarFormatter())
        axes[i].set_title(f'LOG Distribution versus {col}', fontsize=14)
        axes[i].set_ylabel('Percentage (Log Scale %)')
        axes[i].set_xlabel(col)
        
        if i == 0:
            axes[i].legend(title='dvisits', bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            axes[i].get_legend().remove()

    for j in range(i + 1, len(axes)):
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
        
        
        if temp_df[col].nunique() > 10:
            try:
                temp_df[col] = pd.qcut(temp_df[col], q=5, duplicates='drop').astype(str)
            except ValueError:
                temp_df[col] = temp_df[col].astype(str)
        else:
            temp_df[col] = temp_df[col].astype(str)


        sns.barplot(
            data=temp_df, 
            x=col, 
            y='dvisits', 
            ax=axes[idx], 
            palette='magma',
            errorbar=('ci', 95)
        )
        
        axes[idx].set_title(f'divisits mean versus {col}', fontsize=14, weight='bold')
        axes[idx].set_ylabel('dvisits mean')
        axes[idx].set_xlabel(col)
        
        axes[idx].tick_params(axis='x', rotation=45)
        
        media_per_gruppo = temp_df.groupby(col)['dvisits'].mean().sort_values(ascending=False)
        print(f"\ndvisits mean versus {col}:")
        print(media_per_gruppo)

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()