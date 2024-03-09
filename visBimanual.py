import json
import glob
import cv2
import natsort
import numpy as np
from cv2_putText import add_text_to_image
import os
from os.path import join as opj
import sys

from visBoundingBox import get_bboxed_image

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

    
    log=""
    text=""
    for relation in relations:
        oi=relation["object_index"]
        si=relation["subject_index"]

        log = log + _3d[oi]["class_name"]+ " - " + relation["relation_name"] + " - " + _3d[si]["class_name"] +"\n"

        if relation["relation_name"]=="contact":
            
            
            
            # text=text + f"({oi})"+object_mapping[oi] + " - " + f"({si})"+ object_mapping[si] + "\n"
            if oi < si:
                text=text + _3d[oi]["class_name"]+ " - " + _3d[si]["class_name"] +"\n"

    print("#"*20)
    print("#"*20)
    print("FRAME:\n", log, "\n")
    return text

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
    base_dir="/home/enerd/bimanual"

    f_dir="/home/enerd/bimanual/bimacs_rgbd_data/bimacs_rgbd_data"


    frames_path_list=getAllChunks(f_dir, subject, task, take)
    print("len(frames_path_list):", len(frames_path_list))

    # for i in range(len(frames_path_list)):
    frame_idx=0
    while True:

        frame_rgb=frames_path_list[frame_idx]
        # text=getRelations(base_dir, subject, task, take, frame_idx)
        text=""
        left_act, right_act=getActions(base_dir, subject, task, take, frame_idx)

        frame_rgb=cv2.imread(frame_rgb)
        # # # # # # # #  BBOX Starts # # # # # # # #
        BB_DIR=opj(base_dir, "bimacs_derived_data", subject, task, take, "3d_objects", f"frame_{frame_idx}.json")
        with open(BB_DIR) as f:
            objects_3d = json.load(f)

        
        bbox_array=np.zeros((len(objects_3d), 6))

        for obj_idx, obj in enumerate(objects_3d):
            bbox=obj["bounding_box"]
            bbox_array[obj_idx] = np.array([bbox["x0"], bbox["x1"], bbox["y0"], bbox["y1"], bbox["z0"], bbox["z1"]])
            text+= str(obj_idx+1) + " " + obj["class_name"] + "\n"
            # print(obj["class_name"])

        get_bboxed_image(frame_rgb, bbox_array)
        # # # # # # # #  BBOX ENDS  # # # # # # # #

        tone_down=20
        frame_rgb = np.where(frame_rgb < tone_down, 0, frame_rgb-tone_down)

        frame_rgb=cv2.resize(frame_rgb, None, fx=1.5, fy=1.5)

        infograh=np.zeros((frame_rgb.shape[0], 400, 3), dtype=np.uint8)

        cv2.putText(frame_rgb, str(frame_idx), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(frame_rgb, left_act+", " + right_act, (90, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        
        cv2.putText(frame_rgb, subject + " " +  task + " " + take, (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)
        
        infograh=add_text_to_image(infograh, text, bg_color_rgb=(255,255,255))

        combined=np.hstack([frame_rgb, infograh])

        cv2.imshow("combined", combined)
        key=cv2.waitKey(0) & 0xFF

        if key==ord("q"):
            break
        elif key==ord("a"):
            frame_idx=(frame_idx-1)%len(frames_path_list)
        elif key==ord("d"):
            frame_idx=(frame_idx+1)%len(frames_path_list)
        
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # try:
        # main(sys.argv[1], sys.argv[2], sys.argv[3])
    # except:
    main()