# %%
import glob
import pandas as pd
import os
import json
import logging

def get_logger(    
        LOG_FORMAT     = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        LOG_NAME       = '',
        LOG_FILE_INFO  = 'file.log',
        LOG_FILE_ERROR = 'file.err'):

    log           = logging.getLogger(LOG_NAME)
    log_formatter = logging.Formatter(LOG_FORMAT)

    # comment this to suppress console output
    # stream_handler = logging.StreamHandler()
    # stream_handler.setFormatter(log_formatter)
    # log.addHandler(stream_handler)

    file_handler_info = logging.FileHandler(LOG_FILE_INFO, mode='w')
    file_handler_info.setFormatter(log_formatter)
    file_handler_info.setLevel(logging.INFO)
    log.addHandler(file_handler_info)

    file_handler_error = logging.FileHandler(LOG_FILE_ERROR, mode='w')
    file_handler_error.setFormatter(log_formatter)
    file_handler_error.setLevel(logging.ERROR)
    log.addHandler(file_handler_error)

    log.setLevel(logging.INFO)

    return log

if not os.path.exists("/home/joaoneto/biometria/tmp_raw/logs/"):
    print("Creating dir")
    os.makedirs("/home/joaoneto/biometria/tmp_raw/logs", exist_ok=True)




# logging.basicConfig(
#     format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
#     level=logging.INFO,
#     filename="logs/convert_json_to_csv.log",
# )

# logger = logging.getLogger(__name__)

logger = get_logger(LOG_FORMAT="%(asctime)s - %(module)s - %(levelname)s - %(message)s", LOG_FILE_INFO="/home/joaoneto/biometria/tmp_raw/logs/convert_json_to_csv.log",
LOG_FILE_ERROR="/home/joaoneto/biometria/tmp_raw/logs/convert_json_to_csv.err")

for json_file_path in glob.glob('/home/joaoneto/biometria/tmp_raw/sensors/*.json'):
    BASE_PATH_TO_EXPORT = "/home/joaoneto/biometria/tmp_raw/csv_files"
    
    user = json_file_path.split("_")[-2].split("/")[-1]
    logger.info(f"Reading file {json_file_path}")
    filename = json_file_path.split("/")[-1][:-5]
    try:
        with open(json_file_path) as json_data:
            data = json.load(json_data)
    except:
        logger.error(f"Error reading the file {json_file_path}")
        continue

    df = pd.DataFrame(data['accelerometer'])
    df["player_id"] = data["player_id"]

    if not os.path.exists(f"{BASE_PATH_TO_EXPORT}/{user}"):
        logger.info(f"Creating directory {BASE_PATH_TO_EXPORT}/{user}...")
        os.makedirs(f"{BASE_PATH_TO_EXPORT}/{user}", exist_ok=True)
    
    logger.info(f"Exporting file {BASE_PATH_TO_EXPORT}/{user}/{filename}.csv...")
    df.to_csv(f"{BASE_PATH_TO_EXPORT}/{user}/{filename}.csv", index=False)

logger.info("Convertion process has just finished!")

# %%



