import json

notebook_path = 'c:\\Recommendation Engine\\diabetic_iris_classification.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        # Look for the specific plotting cell
        if any('all_val_acc = (h1.history' in line for line in source):
            # Replace the cell's source with a robust version
            new_source = [
                "try:\n",
                "    fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
                "    if 'h1' in globals() and 'h2' in globals() and 'h3' in globals():\n",
                "        # Combine all phases\n",
                "        all_val_acc = (h1.history['val_accuracy'] + h2.history['val_accuracy'] + h3.history['val_accuracy'])\n",
                "        all_val_auc = (h1.history['val_auc'] + h2.history['val_auc'] + h3.history['val_auc'])\n",
                "        all_tr_acc  = (h1.history['accuracy'] + h2.history['accuracy'] + h3.history['accuracy'])\n",
                "        epochs = range(1, len(all_val_acc) + 1)\n",
                "        p1_end = len(h1.history['val_accuracy'])\n",
                "        p2_end = p1_end + len(h2.history['val_accuracy'])\n",
                "    elif 'h3' in globals():\n",
                "        all_val_acc = h3.history['val_accuracy']\n",
                "        all_val_auc = h3.history['val_auc']\n",
                "        all_tr_acc  = h3.history['accuracy']\n",
                "        epochs = range(1, len(all_val_acc) + 1)\n",
                "        p1_end, p2_end = 0, 0\n",
                "    else:\n",
                "        print('No training history found.')\n",
                "        all_val_acc = []\n",
                "        epochs = []\n",
                "\n",
                "    if epochs:\n",
                "        axes[0].plot(epochs, all_tr_acc,  label='Train acc')\n",
                "        axes[0].plot(epochs, all_val_acc, label='Val acc')\n",
                "        if p1_end > 0:\n",
                "            axes[0].axvline(p1_end, color='gray', linestyle='--', alpha=0.5, label='Phase boundary')\n",
                "            axes[0].axvline(p2_end, color='gray', linestyle='--', alpha=0.5)\n",
                "        axes[0].set_title('Accuracy')\n",
                "        axes[0].set_xlabel('Epoch')\n",
                "        axes[0].legend()\n",
                "        axes[0].grid(True, alpha=0.3)\n",
                "\n",
                "        axes[1].plot(epochs, all_val_auc, color='green', label='Val AUC')\n",
                "        if p1_end > 0:\n",
                "            axes[1].axvline(p1_end, color='gray', linestyle='--', alpha=0.5)\n",
                "            axes[1].axvline(p2_end, color='gray', linestyle='--', alpha=0.5)\n",
                "        axes[1].set_title('Validation AUC')\n",
                "        axes[1].set_xlabel('Epoch')\n",
                "        axes[1].legend()\n",
                "        axes[1].grid(True, alpha=0.3)\n",
                "\n",
                "        plt.suptitle('Training History', fontsize=13)\n",
                "        plt.tight_layout()\n",
                "        try:\n",
                "            plt.savefig('/home/ec2-user/SageMaker/training_curves.png', dpi=120)\n",
                "        except Exception:\n",
                "            pass\n",
                "        plt.show()\n",
                "except Exception as e:\n",
                "    print('Error plotting history:', e)\n"
            ]
            cell['source'] = new_source
            print("Successfully fixed the plotting bug in the notebook JSON.")
            break

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
