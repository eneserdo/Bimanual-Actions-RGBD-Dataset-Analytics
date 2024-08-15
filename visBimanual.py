import json
import glob
import cv2
import natsort
import numpy as np
from cv2_putText import add_text_to_image
import os
from os.path import join as opj
import sys

from visBoundingBox import get_2d_bboxed_image, get_bboxed_image

# OBJECT_MAPPING={0: "bowl",
#                 1: "knife",
#                 2: "screwdriver",
#                 3: "cutting_board",
#                 4: "whisk",
#                 5: "hammer",
#                 6: "bottle",
#                 7: "cup",
#                 8: "banana",
#                 9: "cereals",
#                 10: "sponge",
#                 11: "wood",
#                 12: "saw",
#                 13: "hard_drive",
#                 14: "left_hand",
#                 15: "right_hand"}


SUBJECTS = ['subject_{}'.format(i) for i in range(1, 7)]
TASKS= ['task_1_k_cooking', 'task_2_k_cooking_with_bowls', 'task_3_k_pouring',
       'task_4_k_wiping', 'task_5_k_cereals', 'task_6_w_hard_drive',
       'task_7_w_free_hard_drive', 'task_8_w_hammering', 'task_9_w_sawing']
TAKES=['take_{}'.format(take_i) for take_i in range(10)]

ACTIONS = ['idle', 'approach', 'retreat', 'lift', 'place', 'hold', 'pour', 'cut', 'hammer', 'saw',
           'stir', 'screw', 'drink', 'wipe']


REL_SHORTHAND= {"contact": "Cntc",
                "above": "Abv",
                "below": "Blw",
                "left of": "Lo",
                "right of": "Ro",
                "behind of": "Be",
                "in front of": "Fro",
                "inside": "In",
                "surround": "Sur",
                "moving together": "MT",
                "halting together": "HT",
                "fixed moving together": "FMT",
                "getting close": "GC",
                "moving apart": "MA",
                "stable": "S"}

REL_SUBSET= ["contact", 
             "above", 
            #  "below", 
             "left of",
            #  "right of",
             "behind of",
            #  "in front of",
             "inside",
            #  "surround",
             "moving together",
             "halting together",
             "fixed moving together",
             "getting close",
             "moving apart",
            #  "stable"
             ]

def getAllChunks(base_dir, subject, task, take):
    # chunk_parent=base_dir+f"{subject}/{task}/{take}/rgb/"
    chunk_parent=opj(base_dir, subject, task, take, "rgb")

    assert os.path.exists(chunk_parent), f"chunk_parent: {chunk_parent} does not exist"

    chunk_dirs=natsort.natsorted(glob.glob(opj(chunk_parent,"*")))

    frame_path_list=[]
    for chunk in chunk_dirs:
        
        frame_list=natsort.natsorted(glob.glob(opj(chunk, "*")))
        frame_path_list.extend(frame_list)

    return frame_path_list

def getRelations(base_dir, subject, task, take, i):
    # derived_dir=base_dir+"bimacs_derived_data/"
    derived_dir=opj(base_dir, "bimacs_derived_data")

    _3d_objects_dirs=opj(derived_dir, subject, task, take, "3d_objects")
    _3d_objects_json_list=natsort.natsorted(glob.glob(opj(_3d_objects_dirs, "*.json")))

    relation_json_dirs=opj(derived_dir, subject, task, take, "spatial_relations")
    relations_path=natsort.natsorted(glob.glob(opj(relation_json_dirs, "*.json")))


    with open(relations_path[i]) as f:
        relations = json.load(f)

    with open(_3d_objects_json_list[i]) as f:
        _3d = json.load(f)

    
    # log=""
    text_list=[]
    for relation in relations:
        oi=relation["object_index"]
        si=relation["subject_index"]

        # log = log + _3d[oi]["class_name"]+ " - " + relation["relation_name"] + " - " + _3d[si]["class_name"] +"\n"

        # if relation["relation_name"]=="contact":
        if relation["relation_name"] in REL_SUBSET:
        
            # text=text + f"({oi})"+object_mapping[oi] + " - " + f"({si})"+ object_mapping[si] + "\n"
            # if oi < si:
                # text=text + _3d[oi]["class_name"]+ " - " + _3d[si]["class_name"] +"\n"
            # text=text + _3d[oi]["class_name"]+ " - " + _3d[si]["class_name"] +"\n"
            # text_list.append(f"{_3d[oi]['class_name']} - {REL_SHORTHAND[relation['relation_name']]} - {_3d[si]['class_name']}\n")
            text_list.append(f"{oi} - {REL_SHORTHAND[relation['relation_name']]} - {si}")

    # print("#"*20)
    # print("#"*20)
    # print("FRAME:\n", log, "\n")
    text_list = sorted(text_list)
    return text_list


def getActions(base_dir, subject, task, take, i):

    # gt_dirs=base_dir+"bimacs_rgbd_data_ground_truth/"
    # gt_action=gt_dirs+f"{subject}/{task}/{take}.json"
    gt_dirs=opj(base_dir, "bimacs_rgbd_data_ground_truth")
    gt_action=opj(gt_dirs, subject, task, f"{take}.json")

    with open(gt_action) as f:
        actions = json.load(f)


    #right
    for t in range(0, len(actions["right_hand"]) - 1, 2):

        begin = actions["right_hand"][t]
        right_act = actions["right_hand"][t + 1]
        end = actions["right_hand"][t + 2]
    
        if begin <= i < end:
            break
    
    if i == actions["right_hand"][-1]:
        right_act=actions["right_hand"][-2]


    # left
    for t in range(0, len(actions["left_hand"]) - 1, 2):

        begin = actions["left_hand"][t]
        left_act = actions["left_hand"][t + 1]
        end = actions["left_hand"][t + 2]
    
        if begin <= i < end:
            break
    
    if i == actions["left_hand"][-1]:
        left_act=actions["left_hand"][-2]
    

    return ACTIONS[left_act], ACTIONS[right_act]

def main(sub=1, tsk=1, tk=0):
    subject, task, take = SUBJECTS[sub], TASKS[tsk], TAKES[tk]
    print("Reading", subject, task, take)

    base_dir="/home/enerd/bimanual"

    f_dir="/home/enerd/bimanual/bimacs_rgbd_data/bimacs_rgbd_data"
    frames_path_list=getAllChunks(f_dir, subject, task, take)

    frame_idx=0
    MODE=0 # 2D, 2Dfrom3D, 3D

    while True:

        frame_rgb=frames_path_list[frame_idx]
        rel_text_list=getRelations(base_dir, subject, task, take, frame_idx)
        
        left_act, right_act=getActions(base_dir, subject, task, take, frame_idx)

        frame_rgb=cv2.imread(frame_rgb)
        # # # # # # # #  BBOX Starts # # # # # # # #
        if MODE==0:
            BB_DIR=opj(base_dir, "bimacs_derived_data", subject, task, take, "3d_objects", f"frame_{frame_idx}.json")
            obj_text=get_bboxed_image(frame_rgb, BB_DIR)
        elif MODE==1:
            BB_DIR=opj(base_dir, "bimacs_derived_data", subject, task, take, "2d_objects", f"frame_{frame_idx}.json")
            obj_text=get_2d_bboxed_image(frame_rgb, BB_DIR)
        elif MODE==2:
            BB_DIR=opj(base_dir, "bimacs_derived_data", subject, task, take, "3d_objects", f"frame_{frame_idx}.json")
            obj_text=get_bboxed_image(frame_rgb, BB_DIR, is_3d=False)

        # # # # # # # #  BBOX ENDS  # # # # # # # #

        tone_down=20
        frame_rgb = np.where(frame_rgb < tone_down, 0, frame_rgb-tone_down)
        frame_rgb=cv2.resize(frame_rgb, None, fx=1.5, fy=1.5)
        cv2.putText(frame_rgb, str(frame_idx), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(frame_rgb, left_act+", " + right_act, (90, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(frame_rgb, subject + " " +  task + " " + take, (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)


        info_img1=np.zeros((frame_rgb.shape[0], 450, 3), dtype=np.uint8)
        info_img1=add_text_to_image(info_img1, obj_text, bg_color_rgb=(255,255,255), font_scale=0.7)
        
        info_img2=np.ones((frame_rgb.shape[0], 180, 3), dtype=np.uint8)*10
        info_img3=np.ones((frame_rgb.shape[0], 180, 3), dtype=np.uint8)*10

        info_img2=add_text_to_image(info_img2, "\n".join(rel_text_list[:40]), bg_color_rgb=(255,255,255), font_scale=0.5)
        info_img3=add_text_to_image(info_img3, "\n".join(rel_text_list[40:]), bg_color_rgb=(255,255,255), font_scale=0.5)


        combined=np.hstack([frame_rgb, info_img1, info_img2, info_img3])

        cv2.imshow("combined", combined)
        key=cv2.waitKey(0) & 0xFF

        if key==ord("q"):
            break
        elif key==ord("a"):
            frame_idx=(frame_idx-1)%len(frames_path_list)
        elif key==ord("d"):
            frame_idx=(frame_idx+1)%len(frames_path_list)
        elif key==ord("s"):
            MODE = (MODE+1) % 3

        
    cv2.destroyAllWindows()


if __name__ == '__main__':

    if len(sys.argv) == 4:
        main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
    else:
        main()