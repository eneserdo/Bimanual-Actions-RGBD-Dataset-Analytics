import json
import glob
from typing import Tuple
import cv2
import natsort
import numpy as np
import os
from os.path import join as opj
import sys
from collections import Counter
 
SUBJECTS = ['subject_{}'.format(i) for i in range(1, 7)]
TASKS= ['task_1_k_cooking', 'task_2_k_cooking_with_bowls', 'task_3_k_pouring',
       'task_4_k_wiping', 'task_5_k_cereals', 'task_6_w_hard_drive',
       'task_7_w_free_hard_drive', 'task_8_w_hammering', 'task_9_w_sawing']
TAKES=['take_{}'.format(take_i) for take_i in range(10)]

ACTIONS = ['idle', 'approach', 'retreat', 'lift', 'place', 'hold', 'pour', 'cut', 'hammer', 'saw',
           'stir', 'screw', 'drink', 'wipe']

def crawl_dataset():
    for subject in ['subject_{}'.format(i) for i in range(1, 7)]:
        print()
        for task in ['task_1_k_cooking', 'task_2_k_cooking_with_bowls', 'task_3_k_pouring',
                     'task_4_k_wiping', 'task_5_k_cereals', 'task_6_w_hard_drive',
                     'task_7_w_free_hard_drive', 'task_8_w_hammering', 'task_9_w_sawing']:
            print(".", end="", flush=True)
            for take in ['take_{}'.format(take_i) for take_i in range(10)]:
                yield subject, task, take

def both_hand_exists(obj_json) -> Tuple[bool, bool]:

    with open(obj_json) as f:
        _3d = json.load(f)

    obj_list=[]
    for obj in _3d:
        obj_list.append(obj["class_name"])
    
    return ("LeftHand" in obj_list) and ("RightHand" in obj_list) 

def has_null_action(base_dir, subject, task, take, i):

    # gt_dirs=base_dir+"bimacs_rgbd_data_ground_truth/"
    # gt_action=gt_dirs+f"{subject}/{task}/{take}.json"
    gt_dirs=opj(base_dir, "bimacs_rgbd_data_ground_truth")
    gt_action=opj(gt_dirs, subject, task, f"{take}.json")

    with open(gt_action) as f:
        actions = json.load(f)


    # right
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
    

    return left_act==None or right_act==None

def is_empty(obj_json) -> bool:
    with open(obj_json) as f:
        _3d = json.load(f)
        
    return len(_3d)==0

def has_no_edge(rel_json) -> bool:
    with open(rel_json) as f:
        relations = json.load(f)
    
    return len(relations)==0
    


def get_objects(obj_json) -> set:
    
    with open(obj_json) as f:
        _3d = json.load(f)

    obj_list = [obj["class_name"] for obj in _3d]

    s = set(obj_list)
    
    # ignore for now
    # assert len(s)==len(obj_list), "Duplicate objects in frame"
    
    return s


def main():
    base_dir="/home/enerd/bimanual/bimacs_rgbd_data"
    derived_dir=opj(base_dir, "bimacs_derived_data")
    
    TOTAL_CHANGE=0
    TOTAL_FRAME=0
    TOTAL_OBJECTS_COUNT = 0
    DIFFERENT_OBJECTS=Counter()

    for subject, task, take in crawl_dataset():
        
        _3d_objects_dirs=opj(derived_dir, subject, task, take, "3d_objects")
        _3d_objects_json_list=natsort.natsorted(glob.glob(opj(_3d_objects_dirs, "*.json")))

        relation_dirs=opj(derived_dir, subject, task, take, "spatial_relations")
        relations_json_list=natsort.natsorted(glob.glob(opj(relation_dirs, "*.json")))

        prev_obj_set = {}
        
        for i,(obj_json, rel_json) in enumerate(zip(_3d_objects_json_list, relations_json_list)):
        # for i, obj_json in enumerate(_3d_objects_json_list):
            
            if is_empty(obj_json) or has_null_action(base_dir, subject, task, take, i) or has_no_edge(rel_json) or not both_hand_exists(obj_json):
                prev_obj_set = {}
                continue
            
            TOTAL_FRAME += 1

            curr_obj_set = get_objects(obj_json)
            TOTAL_OBJECTS_COUNT += len(curr_obj_set)
            
            if len(prev_obj_set) == 0:
                prev_obj_set = curr_obj_set
                continue
            
            temp=curr_obj_set.symmetric_difference(prev_obj_set)

            DIFFERENT_OBJECTS.update(temp)

            TOTAL_CHANGE += len(temp)

            prev_obj_set = curr_obj_set
            


    # print(f"\n\n{total_frame_count=}, {empty_count=}, {no_edge_count=}, {missing_hand_count=}, {null_count=}")
    print(f"\n{TOTAL_FRAME=}, {TOTAL_OBJECTS_COUNT=}, {TOTAL_CHANGE=}, {DIFFERENT_OBJECTS=}")


if __name__ == '__main__':
    main()