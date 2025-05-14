import os

new_folder = "new12"
os.makedirs(new_folder, exist_ok=True)  # This ensures the folder is created if it doesn't exist


import shutil

files_to_move = ["main.py","train.py", "utils.py", "models.py", "generate_data.py", "curves.py", "load_data.py"]  # List of files to move

for file in files_to_move:
    if os.path.exists(file):
        shutil.copy(file, os.path.join(new_folder, file))
        print(f"Copied {file} to {new_folder}")