# %%
import os
import glob
import shutil

# %%
for json_file_path in glob.glob("/home/joaoneto/biometria/raw/*.json"):
    user_id = json_file_path.split("/")[-1].split("_")[0]

    destination = f"/home/joaoneto/biometria/csv_files"

    if not os.path.exists(destination):
        print(f"Creating directory {destination}")
        os.makedirs(f"{destination}/{user_id}", exist_ok=True)

    print(f"Copying file {json_file_path} to {destination}/{user_id} folder")
    shutil.copy2(json_file_path, f"{destination}/{user_id}")



