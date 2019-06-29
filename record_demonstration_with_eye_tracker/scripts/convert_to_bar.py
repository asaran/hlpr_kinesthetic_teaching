#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np 




n_groups = 2
novice_mean = (16.05793159,17.77138289)
expert_mean = (14.7443329,9.180728071)

novice_std = (1.321735449,5.454284248)
expert_std = (0.8862112544,2.354132062)

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, novice_mean, bar_width, yerr = novice_std,
alpha=0.5,
ecolor='black',
label='Novice',
capsize = 10)

rects2 = plt.bar(index + bar_width, expert_mean, bar_width, yerr = expert_std, 
alpha=0.5,
ecolor='black',
label='Expert',
capsize = 10)


#y = (5,10,15,20)
#plt.hlines(y,0,1.5,linestyles='dashed')
ax.yaxis.grid(True)



plt.xlabel('Tasks')
plt.ylabel('Time')
plt.title('Scores by person')
plt.xticks(index + bar_width, ('Pouring Task','Placement Task'))
plt.legend()

plt.tight_layout()
plt.show()