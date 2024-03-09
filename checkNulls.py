import json
import glob
from typing import Tuple
import cv2
import natsort
import numpy as np
import os
from os.path import join as opj
import sys

 
SUBJECTS = ['subject_{}'.format(i) for i in range(1, 7)]
TASKS= ['task_1_k_cooking', 'task_2_k_cooking_with_bowls', 'task_3_k_pouring',
       'task_4_k_wiping', 'task_5_k_cereals', 'task_6_w_hard_drive',
       'task_7_w_free_hard_drive', 'task_8_w_hammering', 'task_9_w_sawing']
TAKES=['take_{}'.format(take_i) for take_i in range(10)]

ACTIONS = ['idle', 'approach', 'retreat', 'lift', 'place', 'hold', 'pour', 'cut', 'hammer', 'saw',
           'stir', 'screw', 'drink', 'wipe']

def crawl_dataset():
    for subject in ['subject_{}'.format(i) for i in range(1, 7)]:
        for task in ['task_1_k_cooking', 'task_2_k_cooking_with_bowls', 'task_3_k_pouring',
                     'task_4_k_wiping', 'task_5_k_cereals', 'task_6_w_hard_drive',
                     'task_7_w_free_hard_drive', 'task_8_w_hammering', 'task_9_w_sawing']:
            for take in ['take_{}'.format(take_i) for take_i in range(10)]:
                yield subject, task, take


def both_hand_exists(obj_json) -> Tuple[bool, bool]:

    with open(obj_json) as f:
        _3d = json.load(f)

    obj_list=[]
    for obj in _3d:
        obj_list.append(obj["class_name"])
    
    return ("LeftHand" in obj_list), ("RightHand" in obj_list) 

def getActions(base_dir, subject, task, take, i):

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
    

    return left_act, right_act

def is_empty(obj_json) -> bool:
    with open(obj_json) as f:
        _3d = json.load(f)
        
    return len(_3d)==0

def has_no_edge(rel_json) -> bool:
    with open(rel_json) as f:
        relations = json.load(f)
    
    return len(relations)==0

def main():
    base_dir="/home/enerd/bimanual/bimacs_rgbd_data"
    derived_dir=opj(base_dir, "bimacs_derived_data")
    
    missing_hand_count=0
    null_count=0
    total_frame_count=0
    empty_count=0
    no_edge_count=0

    for subject, task, take in crawl_dataset():
        
        _3d_objects_dirs=opj(derived_dir, subject, task, take, "3d_objects")
        _3d_objects_json_list=natsort.natsorted(glob.glob(opj(_3d_objects_dirs, "*.json")))

        relation_dirs=opj(derived_dir, subject, task, take, "spatial_relations")
        relations_json_list=natsort.natsorted(glob.glob(opj(relation_dirs, "*.json")))

        for i,(obj_json, rel_json) in enumerate(zip(_3d_objects_json_list, relations_json_list)):
            total_frame_count+=1
            l_exst, r_exst=both_hand_exists(obj_json) 
            
            left_act, right_act=getActions(base_dir, subject, task, take, i)

            if has_no_edge(rel_json):
                print(f"No edge!, Acts: {left_act}, {right_act}", "dir: ",subject, task, take, "f", i)
                no_edge_count+=1
            # 
            if is_empty(obj_json):
                print(f"Empty graph!, Acts: {left_act}, {right_act}", "dir: ",subject, task, take, "f", i)
                empty_count+=1
            elif not l_exst or not r_exst:
                print(f"Missing: L{int(l_exst)}, R{int(r_exst)}, Acts: {left_act}, {right_act}", "dir: ",subject, task, take, "f", i)
                missing_hand_count+=1
            elif left_act==None or right_act==None:
                print("Exists but null! dir: ", subject, task, take, "f",i)
                null_count+=1


    print(f"\n\n{total_frame_count=}, {empty_count=}, {no_edge_count=}, {missing_hand_count=}, {null_count=}")



if __name__ == '__main__':
    main()