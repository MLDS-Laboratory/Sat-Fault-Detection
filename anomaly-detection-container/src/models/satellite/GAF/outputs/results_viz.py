# results_viz.py
import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns

# 1) Load your CSV
csv_path = "grid_search_results.csv"  # adjust if needed
df = pd.read_csv(csv_path)

# 2) Parse the list-columns into Python lists
list_cols = ['train_loss', 'val_loss', 'train_f1', 'val_f1']
for col in list_cols:
    df[col] = df[col].apply(ast.literal_eval)

# 3) Compute the final (last-epoch) metrics
df['final_train_loss'] = df['train_loss'].apply(lambda lst: lst[-1])
df['final_val_loss']   = df['val_loss'].apply(lambda lst: lst[-1])
df['final_train_f1']   = df['train_f1'].apply(lambda lst: lst[-1])
df['final_val_f1']     = df['val_f1'].apply(lambda lst: lst[-1])

# 4) Identify the best config by test_f1
best_idx    = df['test_f1'].idxmax()
best_config = df.loc[best_idx]

print("\n=== Best hyperparameters (by test_f1) ===")
for k,v in best_config[['model','epochs','batch_size','lr','loss_fn']].items():
    print(f"{k:>12}: {v}")
print("\n=== Metrics for best run ===")
for k,v in best_config[['test_acc','test_f1',
                        'final_train_loss','final_val_loss',
                        'final_train_f1','final_val_f1']].items():
    print(f"{k:>18}: {v:.4f}" if isinstance(v, float) else f"{k:>18}: {v}")

# 5) Plotting
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

# 5a) Bar plot of test_f1
plt.subplot(2,2,1)
sns.barplot(x=df.index, y='test_f1', data=df)
plt.axhline(best_config['test_f1'], color='red', linestyle='--')
plt.title("Test F1 by Configuration")
plt.xlabel("Run index")
plt.ylabel("Test F1")

# 5b) Scatter final_val_loss vs final_val_f1
plt.subplot(2,2,2)
sc = plt.scatter(
    df['final_val_loss'],
    df['final_val_f1'],
    c=df['test_f1'],
    cmap='viridis',
    s=100
)
plt.colorbar(sc, label='Test F1')
plt.scatter(
    best_config['final_val_loss'],
    best_config['final_val_f1'],
    color='red',
    edgecolor='black',
    s=200,
    label='Best run'
)
plt.title("Val Loss vs Val F1")
plt.xlabel("Final Val Loss")
plt.ylabel("Final Val F1")
plt.legend()

# 5c) Learning curves for the best run
plt.subplot(2,1,2)
epochs = list(range(1, len(best_config['train_loss'])+1))
plt.plot(epochs, best_config['train_loss'], label='Train Loss')
plt.plot(epochs, best_config['val_loss'],   label='Val   Loss')
plt.plot(epochs, best_config['train_f1'],   label='Train F1')
plt.plot(epochs, best_config['val_f1'],     label='Val   F1')
plt.axvline(len(epochs), linestyle='--', color='gray')
plt.title("Learning Curves (Best Run)")
plt.xlabel("Epoch")
plt.legend(loc='best')

plt.tight_layout()
plt.show()
