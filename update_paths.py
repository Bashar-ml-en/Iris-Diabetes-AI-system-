import os

script_path = r'c:\Recommendation Engine\extracted_notebook.py'
local_dir = r'c:/Recommendation Engine/'

with open(script_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the AWS cloud paths with the local Windows path
updated_content = content.replace('/home/ec2-user/SageMaker/', local_dir)

with open(script_path, 'w', encoding='utf-8') as f:
    f.write(updated_content)

print("Successfully updated all AWS cloud paths to local Windows paths!")
