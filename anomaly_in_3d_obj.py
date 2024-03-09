"""
    checks for:
    - more than 2 hands
    - changes in object count
"""

import os, json, glob
from natsort import natsorted
from tqdm import tqdm

def crawl_dataset():
    for subject in ['subject_{}'.format(i) for i in range(1, 7)]:
        for task in ['task_1_k_cooking', 'task_2_k_cooking_with_bowls', 'task_3_k_pouring',
                     'task_4_k_wiping', 'task_5_k_cereals', 'task_6_w_hard_drive',
                     'task_7_w_free_hard_drive', 'task_8_w_hammering', 'task_9_w_sawing']:
            for take in ['take_{}'.format(take_i) for take_i in range(10)]:
                yield subject, task, take

root_path="/home/enerd/bimanual/bimacs_derived_data"

missing_hand_counter=0
dublicate_hand_counter=0
object_count_changes=0
total_frame_count=0
# empty_frame_count=0


for subject, task, take in crawl_dataset():
    stt_path=os.path.join(root_path, subject, task, take, "3d_objects")
    frame_paths=natsorted(glob.glob(os.path.join(stt_path, "*")))

    prev_object_count = -1
    mean_object_count=0
    second_moment_object_count=0
    total_frame_count+=len(frame_paths)

    for fp in frame_paths:
        with open(fp) as f:
            objects_3d = json.load(f)
        
        left_hand_count=0
        right_hand_count=0
        object_count=0

        for obj in objects_3d:
            if obj["class_name"]=="LeftHand":
                left_hand_count+=1 
            elif obj["class_name"]=="RightHand":
                right_hand_count+=1
            
            object_count+=1

        if left_hand_count != 1 or right_hand_count !=1:
            # print(f"H Anomaly in {fp}: {left_hand_count} left, {right_hand_count} right")
            ...
        
        if (left_hand_count == 1 and right_hand_count == 0) or (left_hand_count == 0 and right_hand_count == 1):
            missing_hand_counter+=1
        
        if left_hand_count > 1 or right_hand_count > 1:
            dublicate_hand_counter+=1
        
        if object_count != prev_object_count and prev_object_count!=-1:
            # print(f"O Anomaly in {fp}: {prev_object_count} -> {object_count}")
            object_count_changes+=1


        mean_object_count+=object_count
        second_moment_object_count+=object_count**2


        prev_object_count=object_count
    
    mean_object_count/=len(frame_paths)
    second_moment_object_count/=len(frame_paths)
    variance_object_count=second_moment_object_count-mean_object_count**2
    std_object_count=variance_object_count**0.5
    print(f"{fp} m: {mean_object_count:.3f}, std: {std_object_count:.3f}")
    

print(f"Total frame count: {total_frame_count}")
print(f"Missing hand count: {missing_hand_counter}")
print(f"Dublicate hand count: {dublicate_hand_counter}")
print(f"Object count changes: {object_count_changes}")
