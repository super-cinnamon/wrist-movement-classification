import glob
import os

import pandas as pd
import numpy as np
from scipy.stats import mode as sp_mode

from scipy.stats import skew, kurtosis

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')

dataset = pd.DataFrame(columns=["maxroll", "maxpitch", "maxyaw", "minroll", "minpitch", "minyaw", 
                                 "avgroll", "avgpitch", "avgyaw", "ranroll", "ranpitch", "ranyaw", 
                                 "skewroll", "skewpitch", "skewyaw", "moderoll", "modepitch", "modeyaw", 
                                 "medianroll", "medianpitch", "medianyaw", "stdroll", "stdpitch", "stdyaw", 
                                 "kurtosisroll", "kurtosispitch", "kurtosisyaw", "class"])
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
            # add mode, median, standard deviation, kurtosis
            move_mode = sp_mode(roll).mode, sp_mode(pitch).mode, sp_mode(yaw).mode
            move_median = np.median(roll), np.median(pitch), np.median(yaw)
            move_std = np.std(roll), np.std(pitch), np.std(yaw)
            move_kurtosis = kurtosis(roll), kurtosis(pitch), kurtosis(yaw)

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
                "moderoll": move_mode[0],
                "modepitch": move_mode[1],
                "modeyaw": move_mode[2],
                "medianroll": move_median[0],
                "medianpitch": move_median[1],
                "medianyaw": move_median[2],
                "stdroll": move_std[0],
                "stdpitch": move_std[1],
                "stdyaw": move_std[2],
                "kurtosisroll": move_kurtosis[0],
                "kurtosispitch": move_kurtosis[1],
                "kurtosisyaw": move_kurtosis[2],
                "class": class_
            }

            # create a df out of it and concat to full dataset df
            move_row = pd.DataFrame(move_dict, index=[0])
            dataset = pd.concat([dataset, move_row], ignore_index=True)

    return dataset


on_moves = os.path.join(DATA_PATH, "raw_data", "MPUREAD 1 move right*.txt")
off_moves = os.path.join(DATA_PATH, "raw_data", "MPUREAD 1 move wrong*.txt")

on_moves_interns = os.path.join(DATA_PATH, "raw_data", "move right*.txt")
off_moves_interns = os.path.join(DATA_PATH, "raw_data", "move wrong*.txt")

# start with on moves
dataset = process_move(file_schema=on_moves, dataset=dataset, class_=1)
dataset = process_move(file_schema=on_moves_interns, dataset=dataset, class_=1)

# off moves
dataset = process_move(file_schema=off_moves, dataset=dataset, class_=0)
dataset = process_move(file_schema=off_moves_interns, dataset=dataset, class_=0)

dataset.to_csv(os.path.join(DATA_PATH, "processed_dataset.csv"), index=False)


individual_moves = process_move(file_schema=on_moves, dataset=individual_moves, class_=1, get_individual=True)
individual_moves = process_move(file_schema=on_moves_interns, dataset=individual_moves, class_=1, get_individual=True)

individual_moves = process_move(file_schema=off_moves, dataset=individual_moves, class_=0, get_individual=True)
individual_moves = process_move(file_schema=off_moves_interns, dataset=individual_moves, class_=0, get_individual=True)

individual_moves.to_csv(os.path.join(DATA_PATH, "individual_moves.csv"), index=False)
