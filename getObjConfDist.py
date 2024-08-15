# calculate confidence distribution for 3d object bounding boxes (object wise)

import math
import os
import json
import glob
from natsort import natsorted
from tqdm import tqdm
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

def crawl_dataset():
    for subject in ['subject_{}'.format(i) for i in range(1, 7)]:
        for task in ['task_1_k_cooking', 'task_2_k_cooking_with_bowls', 'task_3_k_pouring',
                     'task_4_k_wiping', 'task_5_k_cereals', 'task_6_w_hard_drive',
                     'task_7_w_free_hard_drive', 'task_8_w_hammering', 'task_9_w_sawing']:
            for take in ['take_{}'.format(take_i) for take_i in range(10)]:
                yield subject, task, take

objects = ['bowl', 'knife', 'screwdriver', 'cuttingboard', 'whisk', 'hammer', 'bottle', 'cup',
           'banana', 'cereals', 'sponge', 'woodenwedge', 'saw', 'harddrive', 'LeftHand',
           'RightHand']



if __name__ == "__main__":
    # derived_data_basepath="/home/enerd/bimanual/bimacs_derived_data"

    # conf_dist = {obj: [0]*1001 for obj in objects}
    # iter=0

    # for subject, task, take in crawl_dataset():
    #     stt_path=os.path.join(derived_data_basepath, subject, task, take, "3d_objects")
    #     frame_paths=natsorted(glob.glob(os.path.join(stt_path, "*")))

    #     iter += 1
    #     if iter % 10 == 0:
    #         print(".", end="", flush=True)

    #     for fp in frame_paths:
    #         with open(fp) as f:
    #             objects_3d = json.load(f)
    #         for obj in objects_3d:
    #             class_name=obj["class_name"]
                
    #             certainty=int(obj["certainty"]*1000)

    #             try:
    #                 conf_dist[class_name][certainty]+=1
    #             except Exception as e:
    #                 print("Error: ", class_name, certainty)
    #                 raise e

    with open("conf_dist.json") as f:
        conf_dist = json.load(f)


    # for obj in objects:
    #     plt.plot(conf_dist[obj][:-10], label=obj)
    # plt.legend()
    # plt.show()

    th_200 = 0
    th_150 = 0
    th_100 = 0
    th_50 = 0
    total = 0

    for obj in objects:
        conf_dist[obj] = np.array(conf_dist[obj], dtype=np.float32)

        th_200 += np.sum(conf_dist[obj][:200])
        th_150 += np.sum(conf_dist[obj][:150])
        th_100 += np.sum(conf_dist[obj][:100])
        th_50 += np.sum(conf_dist[obj][:50])

        total += np.sum(conf_dist[obj])

        conf_dist[obj] /= np.sum(conf_dist[obj])
        conf_dist[obj] = conf_dist[obj][:-50] 


    print("th_200: ", th_200, "ratio: ", th_200/total)
    print("th_150: ", th_150, "ratio: ", th_150/total)
    print("th_100: ", th_100, "ratio: ", th_100/total)
    print("th_50: ", th_50, "ratio: ", th_50/total)
    

    fig = px.line(conf_dist)
    fig.show()


    # with open("conf_dist.json", "w") as f:
    #     json.dump(conf_dist, f)

