import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
import seaborn as sns
# Corrected sample dummy dataset creation
data = {
    'Model': ['Vanilla CNN', 'ResNet', 'DenseNet'] * 14 * 2,  # 3 models, repeated for 14 image types and 2 data augmentation conditions
    'ImageType': list(range(1, 15)) * 6,  # 14 image types, repeated for each model
    'DataAugmentation': ['Yes', 'No'] * 42,  # Yes or No for each condition
    'Accuracy': [0.85, 0.88, 0.87, 0.90, 0.86, 0.89, 0.85, 0.87, 0.88, 0.91, 0.86, 0.87, 0.89, 0.85,
                 0.83, 0.86, 0.85, 0.87, 0.84, 0.86, 0.82, 0.84, 0.85, 0.88, 0.83, 0.85, 0.87, 0.84] * 3,  # Repeated to match length
    'Precision': [0.83, 0.87, 0.85, 0.88, 0.84, 0.86, 0.85, 0.88, 0.89, 0.92, 0.85, 0.87, 0.89, 0.84,
                  0.81, 0.85, 0.83, 0.86, 0.82, 0.85, 0.80, 0.83, 0.85, 0.87, 0.82, 0.84, 0.86, 0.83] * 3,  # Repeated to match length
    'Recall': [0.82, 0.86, 0.85, 0.87, 0.84, 0.85, 0.84, 0.86, 0.87, 0.89, 0.85, 0.86, 0.87, 0.83,
               0.80, 0.84, 0.82, 0.85, 0.81, 0.84, 0.79, 0.82, 0.83, 0.85, 0.81, 0.83, 0.85, 0.82] * 3,  # Repeated to match length
    'F1Score': [0.84, 0.87, 0.86, 0.89, 0.85, 0.87, 0.85, 0.87, 0.88, 0.90, 0.86, 0.87, 0.88, 0.84,
                0.82, 0.85, 0.84, 0.86, 0.83, 0.85, 0.81, 0.84, 0.86, 0.87, 0.83, 0.85, 0.86, 0.83] * 3   # Repeated to match length
}
# Convert the data into a pandas DataFrame
df = pd.DataFrame(data)
# Conduct a three-way ANOVA for each performance metric
performance_metrics = ['Accuracy', 'Precision', 'Recall', 'F1Score']
for metric in performance_metrics:
    print(f"\n=== ANOVA Results for {metric} ===")
    # Fit the ANOVA model with interaction effects
    model = ols(f'{metric} ~ C(Model) * C(ImageType) * C(DataAugmentation)', data=df).fit()
    # Perform the ANOVA
    anova_results = anova_lm(model, typ=2)
    print(anova_results)
# Optional: Visualization of the interaction effects
# Example plot of interaction effects for one metric (Accuracy)
plt.figure(figsize=(10, 6))
# Updated to remove 'style' and add 'markers' and 'linestyles'
sns.pointplot(x='ImageType', y='Accuracy', hue='Model', markers=["o", "s", "d"], linestyles=["-", "--", "-."], data=df)
plt.title('Interaction of Model, Image Type, and Data Augmentation on Accuracy')
plt.xlabel('Image Type (KNN Clusters)')
plt.ylabel('Accuracy')
plt.show()
# If you want to plot interaction effects for other metrics, use similar code:
# Example for Precision
plt.figure(figsize=(10, 6))
sns.pointplot(x='ImageType', y='Precision', hue='Model', markers=["o", "s", "d"], linestyles=["-", "--", "-."], data=df)
plt.title('Interaction of Model, Image Type, and Data Augmentation on Precision')
plt.xlabel('Image Type (KNN Clusters)')
plt.ylabel('Precision')
plt.show()
# Example for Recall
plt.figure(figsize=(10, 6))
sns.pointplot(x='ImageType', y='Recall', hue='Model', markers=["o", "s", "d"], linestyles=["-", "--", "-."], data=df)
plt.title('Interaction of Model, Image Type, and Data Augmentation on Recall')
plt.xlabel('Image Type (KNN Clusters)')
plt.ylabel('Recall')
plt.show()
# Example for F1Score
plt.figure(figsize=(10, 6))
sns.pointplot(x='ImageType', y='F1Score', hue='Model', markers=["o", "s", "d"], linestyles=["-", "--", "-."], data=df)
plt.title('Interaction of Model, Image Type, and Data Augmentation on F1Score')
plt.xlabel('Image Type (KNN Clusters)')
plt.ylabel('F1 Score')
plt.show()
