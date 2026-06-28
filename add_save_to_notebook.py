import json

notebook_path = 'c:\\Recommendation Engine\\diabetic_iris_classification.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the last code cell
last_code_cell = None
for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        last_code_cell = cell

if last_code_cell is not None:
    # Append the model save code
    save_code = [
        "\n# --- ADDED BY AI FOR STREAMLIT INTEGRATION ---\n",
        "model.save('stage1_diabetes_classifier.h5')\n",
        "print('Stage 1 Model successfully exported to stage1_diabetes_classifier.h5!')\n"
    ]
    last_code_cell['source'].extend(save_code)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Successfully injected model.save() into the notebook.")
