#!/usr/bin/env python3

# BAR plot for Placement Exp Gaze Analysis

import matplotlib.pyplot as plt
import numpy as np 

my_dpi = 96
n_groups = 2

#KT- experts
# bowl_mean = (7.96, 0.98)
# plate_mean = (9.1, 12.44)
# bowl_std = (2.887504795, 0.3444793042)
# plate_std = (4.98141893, 3.400398964)

# KT - novices
bowl_mean = (8.564832372,2.095708538)
plate_mean = (3.056004211,11.54080015)
bowl_std = (1.895773581,0.8309969122)
plate_std = (1.093722849,1.785922959)

# video demos - all users
# bowl_mean = (27.2, 5.73)
# plate_mean = (3.21, 25.84)
# bowl_std = (6.904718854, 2.373352712)
# plate_std = (1.491990404, 5.949224444)

# create plot
fig, ax = plt.subplots(figsize=(500/my_dpi, 600/my_dpi), dpi=my_dpi)

# index = np.arange(n_groups)
# index = [0, 0.5]
index=[0, 1]
print(index)
bar_width = 0.4
opacity = 0.8
# plt.rcParams.update({'font.size': 18})

rects1 = plt.bar(index, bowl_mean, bar_width, yerr = bowl_std,
alpha=0.5,
ecolor='black',
label='Yellow Bowl Fixations',
capsize = 10,
color = ['orange']) #yellow

rects2 = plt.bar([i + bar_width for i in index], plate_mean, bar_width, yerr = plate_std, 
alpha=0.5,
ecolor='black',
label='Red Plate Fixations',
capsize = 10,
color= ['red'])

#y = (5,10,15,20)
#plt.hlines(y,0,1.5,linestyles='dashed')
ax.yaxis.grid(True)
# plt.ylim([0,42]) # video
# plt.ylim([0,17.5]) # KT - expert
plt.ylim([0,15]) # KT- novices

# plt.xlabel('% time')
# plt.ylabel('% time', size=MEDIUM_SIZE)

# font sizes
SMALL_SIZE = 18
MEDIUM_SIZE = 20
LARGE_SIZE = 20
# plt.ylabel('% time', size=MEDIUM_SIZE)
plt.rc('font', size=MEDIUM_SIZE)
plt.rc('xtick', labelsize=MEDIUM_SIZE)
plt.rc('ytick', labelsize=MEDIUM_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE-2)
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=SMALL_SIZE) 
ax.tick_params(axis='both', which='major', labelsize=SMALL_SIZE)
# ax.tick_params(axis='both', which='minor', labelsize=8)

# plt.title('All Users', fontsize=LARGE_SIZE)

plt.xticks([i + bar_width/2 for i in index], ('Bowl-relative \ninstructions','Plate-relative \ninstructions'))
# plt.legend(loc=0)
ax.legend(loc='upper left')#, bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=False)

plt.tight_layout()
plt.show()

