
# Import modules
from math import sqrt
import pandas as pd
import numpy as np
import csv


# Define landmarks and normalized landmarks
landmark = [6, 168, 8, 193, 222, 224, 223, 113, 226, 25, 24, 22, 112, 243, 155, 153, 144, 23, 161, 160, 159, 157, 173, 246, 247, 27, 30, 56, 122, 231, 230, 229, 52, 53,
            55, 413, 464, 414, 463, 398, 382, 341, 453, 441, 285, 442, 385, 380, 252, 451, 443, 257, 386, 374, 253, 254, 173, 387, 445, 467, 263, 255, 359, 446, 342, 261, 348]

normalizer_lm = [(151, 175), (396, 337), (108, 171)]


# Read CSV file
df = pd.read_csv('initial_dataset.csv')

# Calculate normalization distance for each image based on normalizer_lm

df_original = df
large_empty_list = []
for element in normalizer_lm:

    normalizer_lm_dist = np.sqrt((
        df_original[str(element[0])+'x']-df_original[str(element[1])+'x'])**2+(df_original[str(element[0])+'y']-df_original[str(element[1])+'y'])**2)

    large_empty_list.append(list(normalizer_lm_dist))

norm_dist = np.transpose(np.mean(np.array(large_empty_list), axis=0))


# Generate tuple of (x,y) coordinate  point  for the "landmark" vector

x_coordinate = df[[str(i)+'x' for i in landmark]]
y_coordinate = df[[str(i)+'y' for i in landmark]]


df = pd.DataFrame(np.rec.fromarrays((x_coordinate.values, y_coordinate.values)
                                    ).tolist(), columns=x_coordinate.columns, index=x_coordinate.index)


# Generate pair-wise tuple and calculate the distance

input_dataset = []


def calc_distance(p1, p2):
    return sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)


for row in df.to_numpy():
    test_list = row
    result_list = [(a, b) for idx, a in enumerate(test_list)
                   for b in test_list[idx + 1:]]
    distance_list = [calc_distance(pair[0], pair[1]) for pair in result_list]
    input_dataset.append(distance_list)


input_dataset = np.array(input_dataset)
# print(input_dataset.shape)  # Shape of the dataset

# Normalisng row-wise using python broadcasting

final_dataset = input_dataset/(norm_dist[:, np.newaxis])

final_dataset_df = pd.DataFrame(final_dataset)


# Slicing (from original data) and attaching output coloumn


test = df_original[['output']]
final = final_dataset_df.join(test)

print(final.shape)
