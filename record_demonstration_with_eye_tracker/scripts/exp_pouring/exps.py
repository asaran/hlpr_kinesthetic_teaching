# Experiments for the Pouring Task
# 1. Percentage of time during entire demo - spent on objects, gripper or other parts of workspace 
#		a. Measure differences between novice and experts - video demos
#		b. Measure differences between novice and experts - KT demos
# 2. Perecentage accuarcy to predict reference frame per keyframe (same consecutive keyframes clubbed together) -- NO ANOVA
# 		a. Video versus KT demos (expert users)
# 		b. Video versus KT demos (novice users)		
# 		c. Video demos over time (expert users)
# 		d. Video demos over time (novice users)
# 		e. KT demos over time (expert users)
# 		f. KT demos over time (novice users)


import argparse
from utils import takeClosest
parser = argparse.ArgumentParser()
parser.add_argument("-eid", type=string, default="1a", help='Experiment ID')
args = parser.parse_args()

expert_dir = '/home/asaran/gaze_lfd_ws/src/hlpr_kinesthetic_teaching/record_demonstration_with_eye_tracker/data/pouring/experts/'    
experts = os.listdir(expert_dir)
# print(users)

novice_dir = '/home/asaran/gaze_lfd_ws/src/hlpr_kinesthetic_teaching/record_demonstration_with_eye_tracker/data/pouring/novices/'    
novices = os.listdir(novice_dir)

order = {'KT1':'kv','KT2':'kv','KT3':'vk','KT4':'vk','KT5':'kv','KT6':'kv','KT7':'vk','KT8':'vk','KT9':'kv','KT10':'kv',\
        'KT11':'vk','KT12':'vk','KT13':'kv','KT14':'kv','KT15':'vk','KT16':'vk','KT17':'kv','KT18':'vk','KT19':'vk','KT20':'vk'}

condition_names = {
    'k': 'KT demo',
    'v': 'Video demo'
}


if args.eid == '1a':
	print('Percentage of time during entire demo - spent on objects or other parts of workspace')
	print('Measure differences between novice and experts - video demos')

	# Get all expert users
	print("processing Expert Users' Video Demos...")
	for i in range(len(experts)):
		user = users[i]
	    print(user) #KT1,KT2
	    dir_name = os.listdir(expert_dir+user)

    a = main_dir+user+'/'+dir_name[0]+'/'+'segments/'
    d = os.listdir(a)

    exps = order[user]

    for seg in d:
        print('Segment ', seg)
        demo_type = exps[0] if int(seg)<=3 else exps[1]

        if(int(seg)!=1 and int(seg)!=4):
            continue

        if demo_type!='v':
        	continue

    gp, model, all_vts = read_json(a+seg)
    video_file = my_dir+'fullstream.mp4'
	hsv_timeline, saccade_indices = get_hsv_color_timeline(data, video_file)
    color_name, color_value = get_color_name_from_hist(gaze_coords, img_hsv, radius)


	# One plot showing both novice and expert numbers for objects, other

	exit()

if args.eid == '1b':
	print('Percentage of time during entire demo - spent on objects, gripper or other parts of workspace')
	print('Measure differences between novice and experts - KT demos')
	exit()

if args.eid == '2a':
	print('Perecentage accuarcy to predict reference frame per keyframe')
	print('Video versus KT demos (expert users)')
	exit()

if args.eid == '2b':
	print('Perecentage accuarcy to predict reference frame per keyframe')
	print('Video versus KT demos (novice users)')
	exit()
