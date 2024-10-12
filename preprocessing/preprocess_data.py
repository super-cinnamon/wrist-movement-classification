import glob
import os

import pandas as pd
import numpy as np
from scipy.stats import skew

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')

dataset = pd.DataFrame(columns=["maxroll", "maxpitch", "maxyaw", "minroll", "minpitch", "minyaw", "avgroll", "avgpitch", "avgyaw", "ranroll", "ranpitch", "ranyaw", "skewroll", "skewpitch", "skewyaw", "class"])
individual_moves = pd.DataFrame(columns=["roll", "pitch", "yaw", "class"])


# start with the on class
def process_move(file_schema, dataset, class_, get_individual=False):
    for file_name_wild in glob.glob(file_schema):
        with open(file_name_wild, "r") as file:
            pitch = []
            roll = []
            yaw = []

            for x in file:
                line = x.split("/")
                roll.append(float(line.pop(0)))
                temp = line.pop(0)
                temp.replace("\n", "")
                pitch.append(float(temp))
                temp = line.pop(0)
                temp.replace("\n", "")
                yaw.append(float(temp))
            file.close()

        if get_individual:
            data = {
                "roll": [roll],
                "pitch": [pitch],
                "yaw": [yaw],
                "class": class_
            }
            data_row = pd.DataFrame(data, index=[0])
            dataset = pd.concat([dataset, data_row], ignore_index=True)

        else:
            move_max = np.max(roll), np.max(pitch), np.max(yaw)
            move_min = np.min(roll), np.min(pitch), np.min(yaw)
            move_mean = np.mean(roll), np.mean(pitch), np.mean(yaw)
            move_range = np.max(roll) - np.min(roll), np.max(pitch) - np.min(pitch), np.max(yaw) - np.min(yaw)
            move_skewness = skew(roll), skew(pitch), skew(yaw)

            # make into a dict:
            move_dict = {
                "maxroll": move_max[0],
                "maxpitch": move_max[1],
                "maxyaw": move_max[2],
                "minroll": move_min[0],
                "minpitch": move_min[1],
                "minyaw": move_min[2],
                "avgroll": move_mean[0],
                "avgpitch": move_mean[1],
                "avgyaw": move_mean[2],
                "ranroll": move_range[0],
                "ranpitch": move_range[1],
                "ranyaw": move_range[2],
                "skewroll": move_skewness[0],
                "skewpitch": move_skewness[1],
                "skewyaw": move_skewness[2],
                "class": class_
            }

            # create a df out of it and concat to full dataset df
            move_row = pd.DataFrame(move_dict, index=[0])
            dataset = pd.concat([dataset, move_row], ignore_index=True)

    return dataset


on_moves = os.path.join(DATA_PATH, "raw_data", "MPUREAD 1 move right*.txt")
off_moves = os.path.join(DATA_PATH, "raw_data", "MPUREAD 1 move wrong*.txt")


# start with on moves
dataset = process_move(file_schema=on_moves, dataset=dataset, class_=1)

# off moves
dataset = process_move(file_schema=off_moves, dataset=dataset, class_=0)

dataset.to_csv(os.path.join(DATA_PATH, "processed_dataset.csv"), index=False)


individual_moves = process_move(file_schema=on_moves, dataset=individual_moves, class_=1, get_individual=True)
individual_moves = process_move(file_schema=off_moves, dataset=individual_moves, class_=0, get_individual=True)

individual_moves.to_csv(os.path.join(DATA_PATH, "individual_moves.csv"), index=False)
