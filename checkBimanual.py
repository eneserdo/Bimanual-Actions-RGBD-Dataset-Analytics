"""
Error checking code for "3d_objects" files
Class indexies, class names and instance names are quite incompatible
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

objects = ['bowl', 'knife', 'screwdriver', 'cuttingboard', 'whisk', 'hammer', 'bottle', 'cup',
           'banana', 'cereals', 'sponge', 'woodenwedge', 'saw', 'harddrive', 'LeftHand',
           'RightHand']

derived_data_basepath="/home/enerd/bimanual/bimacs_derived_data"

incompatibility={"class_and_instance":0, "class_and_index": 0, "instance_and_index": 0}
# true_count=0

class_name_index_list=[dict() for i in range(17)]
instance_name_index_list=[dict() for i in range(17)]

def print_all():
    print("\nIncompatibilities")
    print(incompatibility)

    print("\n### class_name_index_list")
    print("Shows in what index which object is encountered based on class name\n")
    for i, e in enumerate(class_name_index_list):
        print(f"index {i}:\n", e)

    print("\n### instance_name_index_list")
    print("Shows in what index which object is encountered based on instance name\n")
    for i, e in enumerate(instance_name_index_list):
        print(f"index {i}:\n", e)

try:
    for subject, task, take in tqdm(crawl_dataset()):
        stt_path=os.path.join(derived_data_basepath, subject, task, take, "3d_objects")
        frame_paths=natsorted(glob.glob(os.path.join(stt_path, "*")))

        for fp in frame_paths:
            with open(fp) as f:
                objects_3d = json.load(f)
            for obj in objects_3d:
                class_name=obj["class_name"]
                instance_name=obj["instance_name"].split("_")[0]
                class_index=obj["class_index"]

                try:
                    class_name_index_list[class_index][class_name]+=1
                except:
                    class_name_index_list[class_index][class_name]=1

                try:
                    instance_name_index_list[class_index][instance_name]+=1
                except:
                    instance_name_index_list[class_index][instance_name]=1


                if class_name!=instance_name:
                    # print("Error1: ", fp.split("/")[7:], ",  ", class_name, " != ", instance_name)
                    incompatibility["class_and_instance"]+=1

                # Both hands have zero index for some reason. This is taken care by the classifier code.
                if class_index==0:
                    continue

                if class_name != objects[class_index-1]:
                    # print("Error2: ", fp.split("/")[7:], ",  ", class_name, " != ", class_index)
                    incompatibility["class_and_index"]+=1

                if instance_name != objects[class_index-1]:
                    # print("Error3: ", fp.split("/")[7:], ",  ", instance_name, " != ",  class_index)
                    incompatibility["instance_and_index"]+=1

                        
except KeyboardInterrupt:
    print("KeyboardInterrupt\n")
    print_all()

finally:
    print("Complete results\n")
    print_all()