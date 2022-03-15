import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

def class_results(true, predictions):
    cf_matrix = confusion_matrix(true, predictions)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ['{0:0.0f}'.format(value) for value in cf_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    clf = classification_report(true, predictions, output_dict=True)
    
    fig, axs = plt.subplots(1, 2, figsize=(16,6))
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues', cbar=False,
                xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'], ax=axs[0])
    axs[0].set_xlabel('Predicted Label', size=15)
    axs[0].set_ylabel('True Label', size=15)
    sns.heatmap(pd.DataFrame(clf).iloc[:-1,:].transpose(), annot=True, cmap='Blues', fmt='.2%',
            yticklabels=['No', 'Yes', 'Accuracy', 'Macro Avg.', 'Weighted Avg'],
            xticklabels=['Precision', 'Recall', 'F1 - Score'], cbar=False, ax=axs[1])
    axs[1].set_title('Classification Report', size=15)
    plt.show()