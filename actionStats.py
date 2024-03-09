import json
import glob
import cv2
import natsort
import numpy as np
# from cv2_putText import add_text_to_image
import os
from os.path import join as opj
import sys
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from torchmetrics.classification import MulticlassF1Score


BASE_DIR="/home/enerd/bimanual/bimacs_rgbd_data"
MAX_ACT_LEN=5000

SUBJECTS = ['subject_{}'.format(i) for i in range(1, 7)]
TASKS= ['task_1_k_cooking', 'task_2_k_cooking_with_bowls', 'task_3_k_pouring',
       'task_4_k_wiping', 'task_5_k_cereals', 'task_6_w_hard_drive',
       'task_7_w_free_hard_drive', 'task_8_w_hammering', 'task_9_w_sawing']
TAKES=['take_{}'.format(take_i) for take_i in range(10)]

ACTIONS = ['idle', 'approach', 'retreat', 'lift', 'place', 'hold', 'pour', 'cut', 'hammer', 'saw',
           'stir', 'screw', 'drink', 'wipe']


def crawl_dataset():
    for subject in ['subject_{}'.format(i) for i in range(1, 7)]:
        # print(".", end="", flush=True)
        for task in ['task_1_k_cooking', 'task_2_k_cooking_with_bowls', 'task_3_k_pouring',
                     'task_4_k_wiping', 'task_5_k_cereals', 'task_6_w_hard_drive',
                     'task_7_w_free_hard_drive', 'task_8_w_hammering', 'task_9_w_sawing']:
            for take in ['take_{}'.format(take_i) for take_i in range(10)]:
                yield subject, task, take


def getActionDistributions():

    action_dist=np.zeros((len(ACTIONS), MAX_ACT_LEN))

    for subject, task, take in crawl_dataset():

        gt_dirs=opj(BASE_DIR, "bimacs_rgbd_data_ground_truth")
        gt_action=opj(gt_dirs, subject, task, f"{take}.json")

        with open(gt_action) as f:
            actions = json.load(f)


        for t in range(0, len(actions["right_hand"])-1, 2):

            begin = actions["right_hand"][t]
            right_act = actions["right_hand"][t + 1]
            end = actions["right_hand"][t + 2]

            if right_act != None:
                action_dist[right_act, end-begin] +=1
        
        
        for t in range(0, len(actions["left_hand"])-1, 2):
            begin = actions["left_hand"][t]
            left_act = actions["left_hand"][t + 1]
            end = actions["left_hand"][t + 2]

            if left_act != None:
                action_dist[left_act, end-begin] +=1


              
    return action_dist




def _get_actions_list(subject, task, take, act_len: int) -> np.ndarray:
    # Returns the action of left hand and right hand, respectively.

    gt_dirs=opj(BASE_DIR, "bimacs_rgbd_data_ground_truth")
    gt_action=opj(gt_dirs, subject, task, f"{take}.json")

    with open(gt_action) as f:
        actions_json = json.load(f)


    actions_list=np.zeros((act_len, 2), dtype=np.int32)

    # right
    for t in range(0, len(actions_json["right_hand"]) - 1, 2):

        begin = actions_json["right_hand"][t]
        right_act = actions_json["right_hand"][t + 1]
        end = actions_json["right_hand"][t + 2]

        right_act = -1 if right_act == None else right_act

        actions_list[begin:end, 1]=right_act

    # left
    for t in range(0, len(actions_json["left_hand"]) - 1, 2):

        begin = actions_json["left_hand"][t]
        left_act = actions_json["left_hand"][t + 1]
        end = actions_json["left_hand"][t + 2]

        left_act = -1 if left_act == None else left_act
        actions_list[begin:end, 0]=left_act
    
    return actions_list


def getActions(subject, task, take, i):
    
    gt_dirs=opj(BASE_DIR, "bimacs_rgbd_data_ground_truth")
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
    

    return left_act, right_act


f1_metric_middle=MulticlassF1Score(num_classes=len(ACTIONS), average='macro')
f1_metric_last=MulticlassF1Score(num_classes=len(ACTIONS), average='macro')

def getActionDiversityDistForN(N):

    diversity_dist=np.zeros((N+1))
    left_right_cooccurance=np.zeros((len(ACTIONS), len(ACTIONS)))
    correct_match_middle=0
    correct_match_last=0
    TOTAL=0
    for subject, task, take in crawl_dataset():


        gt_dirs=opj(BASE_DIR, "bimacs_rgbd_data_ground_truth")
        gt_action=opj(gt_dirs, subject, task, f"{take}.json")
        
        with open(gt_action) as f:
            actions = json.load(f)


        assert actions["right_hand"][-1] == actions["left_hand"][-1]
        T=actions["right_hand"][-1]

        continue_flag=False
        act_list=_get_actions_list(subject, task, take, T)

        for t in range(N, T):
            
            temp_act_seq=np.zeros((2, N))
            # read N action labels

            for i in range(N):
                left_act, right_act = act_list[t-i].tolist()
                # left_act, right_act = getActions(subject, task, take, t-i)
                
                if left_act == None or right_act == None:
                    continue_flag=True
                    break

                temp_act_seq[0, i]=left_act
                temp_act_seq[1, i]=right_act

            if continue_flag:
                continue_flag=False
                continue

            # uniques=np.unique(temp_act_seq[0])
            # diversity_dist[uniques.shape[0]]+=1

            # uniques=np.unique(temp_act_seq[1])
            # diversity_dist[uniques.shape[0]]+=1
            
            # compare action matching for last-pick and middle-pick
            last_pick_left, last_pick_right = temp_act_seq[0, -1], temp_act_seq[1, -1]
            middle_pick_left, middle_pick_right = temp_act_seq[0, N//2], temp_act_seq[1, N//2]

            # correct_match_middle = correct_match_middle + (temp_act_seq[0] == middle_pick_left).sum() + (temp_act_seq[1] == middle_pick_right).sum()
            # correct_match_last = correct_match_last + (temp_act_seq[0] == last_pick_left).sum() + (temp_act_seq[1] == last_pick_right).sum()
            # TOTAL+=2*N

            # f1_metric_middle(torch.tensor(temp_act_seq[0]), torch.ones((N))*middle_pick_left)
            # f1_metric_middle(torch.tensor(temp_act_seq[1]), torch.ones((N))*middle_pick_right)

            # f1_metric_last(torch.tensor(temp_act_seq[0]), torch.ones((N))*last_pick_left)
            # f1_metric_last(torch.tensor(temp_act_seq[1]), torch.ones((N))*last_pick_right)

            left_right_cooccurance[int(last_pick_left), int(last_pick_right)]+=1


    # return diversity_dist, correct_match_last/TOTAL, correct_match_middle/TOTAL, left_right_cooccurance
    return left_right_cooccurance


def main():
    
    # print("getting action distributions...")
    # act_dist=getActionDistributions()


    # # plot action distribution, in separate subplots
    # # plt.figure()
    # # plt.suptitle("Action Distribution")
    # for i, action in enumerate(ACTIONS):
    #     # plt.subplot(4, 4, i+1)
    #     # plt.title(action)
    #     act_dist[i]=act_dist[i]/np.sum(act_dist[i])
    #     mean=np.sum(act_dist[i]*np.arange(MAX_ACT_LEN))

    #     print(f"{action}: mean: {np.round(mean, 2)}, std: {np.round(np.sqrt(np.sum(act_dist[i]*(np.arange(MAX_ACT_LEN)**2)) - mean**2), 2)}")
    #     # print(f"{action}: mean: {np.round(np.mean(var), 2)}, std: {np.round(np.std(var), 2)}")
    #     # plt.xlabel("Action Length")
    #     # plt.ylabel("Frequency")
    #     # plt.plot(act_dist[i]/np.sum(act_dist[i]))

    # # # plt.tight_layout()
    # # # plt.show()
    # # # plt.savefig("action_dist.png")

    print("getting action diversity distributions...")
    div_dists=[]
    Ns=[10, 20, 30, 40]
    for n in Ns:  
        # div, last_pick_acc, middle_pick_acc, lr_cooccurance = getActionDiversityDistForN(n)
        lr_cooccurance = getActionDiversityDistForN(n)
        lr_cooccurance = lr_cooccurance+ lr_cooccurance.T
        # lr_cooccurance = lr_cooccurance/np.sum(lr_cooccurance, axis=1, keepdims=True)
        
        # div_dists.append(div)
        # print(f"N={n} is {last_pick_acc=}, {middle_pick_acc=}")
        # print(np.round(div_dists[-1]/np.sum(div_dists[-1]), 3))
        # print(f"middle pick f1 score: {f1_metric_middle.compute()}")
        # print(f"last pick f1 score: {f1_metric_last.compute()}")

        print("...", flush=True)
        # plot left-right cooccurance
        plt.figure()
        plt.title(f"Left-Right Cooccurance Matrix for N={n}")
        plt.xlabel("Right Hand")
        plt.ylabel("Left Hand")
        plt.xticks(np.arange(len(ACTIONS)), ACTIONS, rotation=90)
        plt.yticks(np.arange(len(ACTIONS)), ACTIONS)
        plt.imshow(np.log(lr_cooccurance+1))
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f"lr_cooccurance_N={n}.png")
        f1_metric_middle.reset()
        f1_metric_last.reset()
        print("", flush=True)



    # # plot diversity distribution
    # plt.figure()
    # plt.title("Action Diversity Distribution")
    # plt.xlabel("Number of Unique Actions")
    # plt.ylabel("Frequency")
    # for i, n in enumerate(Ns):
    #     plt.plot(div_dists[i]/np.sum(div_dists[i]), label=f"N={n}")
    # plt.legend()
    # plt.savefig("action_div_dist.png")




if __name__ == '__main__':
    main()