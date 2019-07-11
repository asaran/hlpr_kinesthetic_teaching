#!/usr/bin/env python3

# BAR plot for Basic Fixation times

import matplotlib.pyplot as plt
import numpy as np 

my_dpi = 96
n_groups = 2

# Video pouring
relevant_mean = (48.83133789, 58.44747335)
irrelevant_mean = (17.0735827, 3.568722724)

relevant_std = (4.719798207, 1.900581741)
irrelevant_std = (3.743707028, 0.6523259221)

# KT pouring
# relevant_mean = (52.22035606, 62.92070108)
# irrelevant_mean = (9.237936545, 6.815472802)

# relevant_std = (1.569479265, 1.834404179)
# irrelevant_std = (1.099233027, 0.8795993263)
					
						
# create plot
fig, ax = plt.subplots(figsize=(600/my_dpi, 600/my_dpi), dpi=my_dpi)
# index = np.arange(n_groups)
# index = [0, 0.5]
index=[0, 1]
print(index)
bar_width = 0.3
opacity = 0.8
# plt.rcParams.update({'font.size': 18})

rects1 = plt.bar(index, relevant_mean, bar_width, yerr = relevant_std,
alpha=0.5,
ecolor='black',
label='Task Relevant Objects',
capsize = 10)

rects2 = plt.bar([i + bar_width for i in index], irrelevant_mean, bar_width, yerr = irrelevant_std, 
alpha=0.5,
ecolor='black',
label='Task Irrelevant Objects',
capsize = 10)

#y = (5,10,15,20)
#plt.hlines(y,0,1.5,linestyles='dashed')
ax.yaxis.grid(True)
# plt.ylim([0,82]) # KT
plt.ylim([0,77]) # video

# plt.xlabel('Tasks')
# plt.ylabel('Time')

# font sizes
SMALL_SIZE = 18
MEDIUM_SIZE = 20
LARGE_SIZE = 20
plt.ylabel('% time', size=MEDIUM_SIZE)
plt.rc('font', size=MEDIUM_SIZE)
plt.rc('xtick', labelsize=MEDIUM_SIZE)
plt.rc('ytick', labelsize=MEDIUM_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=SMALL_SIZE) 
ax.tick_params(axis='both', which='major', labelsize=SMALL_SIZE)
# ax.tick_params(axis='both', which='minor', labelsize=8)

# plt.title('Kinesthetic Demonstrations', fontsize=LARGE_SIZE)
plt.title('Video Demonstrations', fontsize=LARGE_SIZE)

plt.xticks([i + bar_width/2 for i in index], ('Novice Users','Expert Users'))
plt.legend(loc=2)

plt.tight_layout()
plt.show()