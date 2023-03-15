## Libraries Used
import json
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

from glob import glob
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from functools import partial

from asl.utils import natural_keys
import tensorflow as tf

# train.csv file
df = pd.read_csv("data/train.csv")

def add_path(row):
    return "data/"+row.path

df["path"] = df.apply(lambda row: add_path(row), axis=1)
print(df.head())

num_samples = 500
FPS = 10
num_tfrecords = len(df) // num_samples
skf = StratifiedKFold(n_splits=num_tfrecords, random_state=42, shuffle=True)

for i, (_, test_index) in enumerate(skf.split(df.path.values, df.sign.values)):
    break
print("Text Index: ", test_index)


def get_hand_points(hand):
    x = [[hand.iloc[0].x, hand.iloc[1].x, hand.iloc[2].x, hand.iloc[3].x, hand.iloc[4].x], # Thumb
         [hand.iloc[5].x, hand.iloc[6].x, hand.iloc[7].x, hand.iloc[8].x], # Index
         [hand.iloc[9].x, hand.iloc[10].x, hand.iloc[11].x, hand.iloc[12].x], 
         [hand.iloc[13].x, hand.iloc[14].x, hand.iloc[15].x, hand.iloc[16].x], 
         [hand.iloc[17].x, hand.iloc[18].x, hand.iloc[19].x, hand.iloc[20].x], 
         [hand.iloc[0].x, hand.iloc[5].x, hand.iloc[9].x, hand.iloc[13].x, hand.iloc[17].x, hand.iloc[0].x]]

    y = [[hand.iloc[0].y, hand.iloc[1].y, hand.iloc[2].y, hand.iloc[3].y, hand.iloc[4].y],  #Thumb
         [hand.iloc[5].y, hand.iloc[6].y, hand.iloc[7].y, hand.iloc[8].y], # Index
         [hand.iloc[9].y, hand.iloc[10].y, hand.iloc[11].y, hand.iloc[12].y],
         [hand.iloc[13].y, hand.iloc[14].y, hand.iloc[15].y, hand.iloc[16].y],
         [hand.iloc[17].y, hand.iloc[18].y, hand.iloc[19].y, hand.iloc[20].y],
         [hand.iloc[0].y, hand.iloc[5].y, hand.iloc[9].y, hand.iloc[13].y, hand.iloc[17].y, hand.iloc[0].y]] 
    return x, y


def get_pose_points(pose):
    x = [[pose.iloc[8].x, pose.iloc[6].x, pose.iloc[5].x, pose.iloc[4].x, pose.iloc[0].x, pose.iloc[1].x, pose.iloc[2].x, pose.iloc[3].x, pose.iloc[7].x], 
         [pose.iloc[10].x, pose.iloc[9].x], 
         [pose.iloc[22].x, pose.iloc[16].x, pose.iloc[20].x, pose.iloc[18].x, pose.iloc[16].x, pose.iloc[14].x, pose.iloc[12].x, 
          pose.iloc[11].x, pose.iloc[13].x, pose.iloc[15].x, pose.iloc[17].x, pose.iloc[19].x, pose.iloc[15].x, pose.iloc[21].x], 
         [pose.iloc[12].x, pose.iloc[24].x, pose.iloc[26].x, pose.iloc[28].x, pose.iloc[30].x, pose.iloc[32].x, pose.iloc[28].x], 
         [pose.iloc[11].x, pose.iloc[23].x, pose.iloc[25].x, pose.iloc[27].x, pose.iloc[29].x, pose.iloc[31].x, pose.iloc[27].x], 
         [pose.iloc[24].x, pose.iloc[23].x]
        ]

    y = [[pose.iloc[8].y, pose.iloc[6].y, pose.iloc[5].y, pose.iloc[4].y, pose.iloc[0].y, pose.iloc[1].y, pose.iloc[2].y, pose.iloc[3].y, pose.iloc[7].y], 
         [pose.iloc[10].y, pose.iloc[9].y], 
         [pose.iloc[22].y, pose.iloc[16].y, pose.iloc[20].y, pose.iloc[18].y, pose.iloc[16].y, pose.iloc[14].y, pose.iloc[12].y, 
          pose.iloc[11].y, pose.iloc[13].y, pose.iloc[15].y, pose.iloc[17].y, pose.iloc[19].y, pose.iloc[15].y, pose.iloc[21].y], 
         [pose.iloc[12].y, pose.iloc[24].y, pose.iloc[26].y, pose.iloc[28].y, pose.iloc[30].y, pose.iloc[32].y, pose.iloc[28].y], 
         [pose.iloc[11].y, pose.iloc[23].y, pose.iloc[25].y, pose.iloc[27].y, pose.iloc[29].y, pose.iloc[31].y, pose.iloc[27].y], 
         [pose.iloc[24].y, pose.iloc[23].y]
        ]
    return x, y


def animation_frame(f, sign, face=True, pose=True, hands=True):
    assert face or pose or hands
    ax.clear()
    
    frame = sign[sign.frame==f]
    
    # Hands
    if hands:
        left = frame[frame.type=='left_hand']
        right = frame[frame.type=='right_hand']
        lx, ly = get_hand_points(left)
        rx, ry = get_hand_points(right)

        for i in range(len(lx)):
            ax.plot(lx[i], ly[i])
        for i in range(len(rx)):
            ax.plot(rx[i], ry[i])
    
    # Pose
    if pose:
        pose = frame[frame.type=='pose']
        px, py = get_pose_points(pose)

        for i in range(len(px)):
            ax.plot(px[i], py[i])
    
    # Face
    if face:
        face = frame[frame.type=='face'][['x', 'y']].values
        
        ax.plot(face[:,0], face[:,1], '.')


run = wandb.init(
    project="kaggle-asl-eda",
    job_type="animation",
)

animation_table = wandb.Table(columns=["path", "animation", "sign", "participant_id"])

for idx in tqdm(test_index):
    tmp_df = df.iloc[idx]

    sign = pd.read_parquet(tmp_df.path)
    sign.y = sign.y * -1 # so that the human is not upside down.

    # Resize sign here.
    # sign_resized = sign[["x", "y", "z"]].values
    # sign_resized = tf.

    fig, ax = plt.subplots()
    l, = ax.plot([], [])
    anim = FuncAnimation(
        fig,
        func=partial(animation_frame, sign=sign),
        frames=sign.frame.unique()
    )

    f = f"eda/video_{idx}.mp4"
    writervideo = animation.FFMpegWriter(fps=FPS)
    anim.save(f, writer=writervideo)

    animation_table.add_data(
        tmp_df.path,
        wandb.Video(f, fps=FPS),
        tmp_df.sign,
        tmp_df.participant_id,
    )

wandb.log({
    "animation": animation_table,
})
