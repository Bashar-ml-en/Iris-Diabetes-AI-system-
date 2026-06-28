import json
import os

notebook_path = 'c:\\Recommendation Engine\\diabetic_iris_classification.ipynb'
script_path = 'c:\\Recommendation Engine\\extracted_notebook.py'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

code_lines = []
for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        for line in source:
            # Comment out lines starting with ! (jupyter magics)
            if line.strip().startswith('!'):
                code_lines.append('# ' + line)
            elif line.strip().startswith('%'):
                code_lines.append('# ' + line)
            else:
                code_lines.append(line)
        code_lines.append('\n')

with open(script_path, 'w', encoding='utf-8') as f:
    f.writelines(code_lines)

print(f"Notebook code extracted to {script_path}. Attempting to run it now...")
