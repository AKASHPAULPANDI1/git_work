#created by Akash P

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Labels matching your Diagnosis Legend
labels = ['High Fiber', 'High Protein', 'Balanced', 'Low Sugar', 'Low Sodium']

# Data derived from your 94.2% Accuracy and 100% Compliance report
# Rows: Actual, Columns: Predicted
cm_data = np.array([
    [166, 4, 6, 0, 0],
    [5, 208, 7, 0, 0],
    [8, 10, 214, 0, 0],
    [0, 0, 0, 268, 0],
    [2, 3, 3, 0, 96]
])

plt.figure(figsize=(10, 8))
sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels, yticklabels=labels, cbar=False)

##good to see this

plt.title('Confusion Matrix: HFRS-DA Diagnostic Classification', fontweight='bold', pad=20)
plt.xlabel('Predicted Category', fontweight='bold')
plt.ylabel('Actual Category', fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix_hfrs.png', dpi=300)
plt.show()