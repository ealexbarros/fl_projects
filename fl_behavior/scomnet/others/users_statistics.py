# %%
import glob
import pandas as pd

CSV_FILES_PATH = "/home/joaoneto/biometria/csv_files/*/*.csv"
USERS_STATISTICS_EXPORT_PATH = "/home/joaoneto/biometria/SCOMNet-1/scomnet/users_statistics.csv"

user_data = pd.concat([pd.read_csv(json_file_path) for json_file_path in glob.glob(CSV_FILES_PATH)])
users_statistics = user_data["player_id"].value_counts().rename("nrows").to_frame()
users_statistics.index.name = "player_id"
users_statistics.to_csv(USERS_STATISTICS_EXPORT_PATH)
