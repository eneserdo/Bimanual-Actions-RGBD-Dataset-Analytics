import matplotlib.pyplot as plt

# # mean and standard deviation values for the length of actions
# means = [10, 15, 20, 25, 30]
# stds = [1, 2, 3, 4, 5]

# # plot means and standard deviations
# means=[ 42.43, 13.27, 10.06, 16.52, 23.38, 86.83, 45.68, 66.1, 59.25, 126.11, 314.77, 128.42, 44.12, 190.27]
# names=[ "idle",  "approach",  "retreat",  "lift",  "place",  "hold",  "pour",  "cut",  "hammer",  "saw",  "stir",  "screw",  "drink",  "wipe"]
# stds= [ 27.48, 7.9, 7.3, 9.56, 11.24, 95.28, 26.12, 30.72, 26.29, 62.25, 144.5, 52.37, 28.25, 51.21]


# plt.errorbar(range(len(means)), means, yerr=stds, fmt='o')
# plt.xticks(range(len(means)), names, rotation=45)
# plt.ylabel('Uzunluk')
# plt.title('Ortalama Aksiyon Uzunlukları')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# N=10 is last_pick_acc=0.8873423056325238, middle_pick_acc=0.9346610939064428
# [0.    0.784 0.196 0.019 0.001 0.    0.    0.    0.    0.    0.   ]


# N=20 is last_pick_acc=0.7854003344082169, middle_pick_acc=0.8730360683148215
# [0.    0.616 0.284 0.078 0.02  0.001 0.    0.    0.    0.    0.    0.
#  0.    0.    0.    0.    0.    0.    0.    0.    0.   ]

# N=30 is last_pick_acc=0.7100549248461369, middle_pick_acc=0.8182084038283925
# [0.    0.513 0.274 0.152 0.05  0.01  0.    0.    0.    0.    0.    0.
#  0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#  0.    0.    0.    0.    0.    0.    0.   ]

# N=40 is last_pick_acc=0.6552441374272688, middle_pick_acc=0.7715440417117957
# [0.    0.441 0.239 0.207 0.083 0.03  0.    0.    0.    0.    0.    0.
#  0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#  0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#  0.    0.    0.    0.    0.   ]

# PLot the probability of action counts

N_10=[0.784, 0.196, 0.019, 0.001, 0, 0]
N_20=[0.616, 0.284, 0.078, 0.02, 0.001, 0]
N_30=[0.513, 0.274, 0.152, 0.05, 0.01, 0]
N_40=[0.441, 0.239, 0.207, 0.083, 0.03, 0]

# plt.figure(figsize=(20, 10))
plt.xlabel("Aksiyon Sayısı")
plt.xticks(range(6), ["1", "2", "3", "4", "5", "6"])
plt.ylabel("Olasılık")
plt.grid(True)
plt.plot(N_10, label="N=10")
plt.plot(N_20, label="N=20")
plt.plot(N_30, label="N=30")
plt.plot(N_40, label="N=40")
plt.tight_layout()
plt.legend()
plt.show()




