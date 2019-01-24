#!/usr/bin/env python2

'''This code tests out using rbfs around each object. The correct reward is to place it to the left of a the spoon (object 0)'''

import random
import numpy as np
import birl
import active_utils as autils
import matplotlib.pyplot as plt
import active_var_complexreward as active_var
import sys

from copy import deepcopy

def test_placements(true_reward, num_test):
    test_rbfs = []
    for i in range(num_test):
        #generate new centers, but keep weights
        num_objs = true_reward.num_objects
        new_centers = np.random.rand(num_objs, 2)
        obj_weights = true_reward.obj_weights.copy()
        abs_weights = true_reward.abs_weights.copy()
        new_rbf = autils.RbfComplexReward(new_centers, obj_weights, abs_weights)
        test_rbfs.append(new_rbf)
        #autils.visualize_reward(new_rbf, "test placement")
    return test_rbfs


#return mean and standard deviation of loss over test placements
def calc_test_reward_loss(test_placements, map_params, visualize=False):
    losses = []
    for placement in test_placements:
        test_config = placement.obj_centers
        true_params = (placement.obj_weights, placement.abs_weights)
        ploss = active_var.calculate_policy_loss(test_config, true_params, map_params)
        if visualize:
            test_map = autils.RbfComplexReward(test_config, map_params[0], map_params[1])
            autils.visualize_reward(test_map, "testing with map reward")
            plt.show()
        losses.append(ploss)
    losses = np.array(losses)
    return np.mean(losses), np.std(losses), np.max(losses)


def calc_test_placement_loss(test_placements, map_params, visualize=False):
    losses = []
    cnt = 0
    for placement in test_placements:
        #print cnt
        cnt += 1
        test_config = placement.obj_centers
        true_params = (placement.obj_weights, placement.abs_weights)
        ploss = active_var.calculate_placement_loss(test_config, true_params, map_params)
        if visualize:
            test_map = autils.RbfComplexReward(test_config, map_params[0], map_params[1])
            autils.visualize_reward(test_map, "testing with map reward")
            plt.show()
        losses.append(ploss)
    losses = np.array(losses)
    return np.mean(losses), np.std(losses), np.max(losses)


if __name__=="__main__":

    rand_seed = 12345
    np.random.seed(rand_seed)

    beta=100.0
    num_steps = 1000
    step_std = 0.05
    burn = 0
    skip = 25
    num_test = 100

    exp = 'plate' # or 'bowl'
    demo_type = 'KT' # 'video' or 'KT'

    num_objects = 2 #plate and bowl
    #object weights are for center, top left, top right, bottom left, bottom right
    if(demo_type=='video'):
        if exp=='bowl':
            obj1_weights = np.array([0.0, 0.5, 0.0, 0.5, 0.0]) #bowl: equal weight on top left and bottom left rbf results in placement directly to left of object
            obj0_weights = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) #plate

            demo_plate = [[0.437,0.668],[0.421,0.665],[0.428,0.459],[0.436,0.77], [0.425,0.777]]
            demo_bowl = [[0.588,0.605],[0.587,0.609],[0.591,0.396],[0.598, 0.711], [0.594,0.724]]
            demo_spoon = [[0.552,0.663],[0.536,0.638],[0.551,0.42], [0.532,0.738], [0.549,0.731]]
            demo_gaze = [[6.854,13.306],[1.96,42.156],[0.0,49.253], [0.0,24.58], [0.0,13.978]]
        #obj2_weights = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) #distractor
        elif exp=='plate':
            obj1_weights = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) #bowl: equal weight on top left and bottom left rbf results in placement directly to left of object
            obj0_weights = np.array([0.0, 0.0, 0.5, 0.0, 0.5]) #plate

            demo_plate = [[0.438,0.670],[0.428,0.657],[0.427,0.461],[0.435,0.761],[0.423,0.777]]
            demo_bowl = [[0.591,0.611],[0.582,0.604],[0.508,0.435],[0.598,0.715],[0.593,0.720]]
            demo_spoon = [[0.507, 0.638],[0.514,0.641],[0.508,0.435],[0.532,0.738],[0.495,0.746]]
            demo_gaze = [[23.42,5.948],[10.577,6.73],[13.861,0.0],[20.382,0.0],[30.0,0.0]]

    if(demo_type=='KT'):
        if exp=='bowl':

            obj1_weights = np.array([0.0, 0.0, 0.5, 0.0, 0.5]) #bowl: equal weight on top left and bottom left rbf results in placement directly to left of object
            obj0_weights = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) #plate

            demo_plate = [[0.574,0.685],[0.554,0.683],[0.552,0.7],[0.533,0.694],[0.566,0.683]] #[0.569,0.683]
            demo_bowl = [[0.353,0.729],[0.339,0.719],[0.317,0.759],[0.349,0.743],[0.333,0.719]] #[0.338,0.728]
            demo_spoon = [[0.448,0.783],[0.434,0.683],[0.398,0.765],[0.424,0.735],[0.427,0.755]] #[0.418,0.704]
            demo_gaze = [[1.353,4.963],[0,5.847],[0.643,11.568],[0.393,21.335],[6.406,12.278]] #[0.888,7.988]
        #obj2_weights = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) #distractor
        elif exp=='plate':
            obj1_weights = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) #bowl: equal weight on top left and bottom left rbf results in placement directly to left of object
            obj0_weights = np.array([0.0, 0.5, 0.0, 0.5, 0.0]) #plate

            demo_plate = [[0.568,0.687],[0.58,0.683],[0.554,0.698],[0.574,0.706],[0.585,0.693]] #[0.571,0.687]
            demo_bowl = [[0.352,0.733],[0.355,0.720],[0.321,0.757],[0.351,0.752],[0.351,0.713]] #[0.343,0.724]
            demo_spoon = [[0.460,0.694],[0.491,0.709],[0.419,0.726],[0.473,0.754],[0.485,0.715]] #[0.473,0.704]
            demo_gaze = [[14.286,0.672],[11.538,0],[8.128,6.676],[26.466,10.977],[9.191,0]] #[20.0,32.174]

    obj_weights = np.concatenate((obj0_weights, obj1_weights))
    abs_weights = np.array([0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0]) # no absolute placement preferences
    
    num_obj_weights = len(obj_weights)
    num_abs_weights = len(abs_weights)

    birl = birl.BIRL(num_obj_weights, num_abs_weights, beta, num_steps, step_std, burn, skip)

    #give three demos in different positions
    num_demos = 5


    for i in range(num_demos):
    # for i in [2,3,4]:
        #generate random object placements
        # obj_centers += np.random.random((num_objects,2))*0.01
        # demo_rbf = autils.RbfComplexReward(obj_centers, obj_weights, abs_weights)
        # best_x, reward = demo_rbf.estimate_best_placement()
        # best_x = np.array([0.5, 0.5]) + np.random.random(2)*0.1

        obj_centers = np.array([demo_plate[i],demo_bowl[i]])
        look_times = np.array(demo_gaze[i])
        best_x = np.array(demo_spoon[i]) 
        true_rbf = autils.RbfComplexReward(obj_centers, obj_weights, abs_weights)

        birl.add_gaze_demonstration(obj_centers, best_x, look_times)

        print "demo", best_x
        autils.visualize_reward(true_rbf, "demo {} and ground truth reward".format(i) )
    #plt.show()


    #run birl to get MAP estimate
    birl.run_inference(gaze=True)
    # birl.run_gaze_inference()


    #print out the map reward weights
    map_obj_wts, map_abs_wts = birl.get_map_params()
    # print "obj weights", map_obj_wts
    # print "abs weights", map_abs_wts

    # test_file = "./data/user_study_gaze" + str(rand_seed) + "_random.txt"
    test_file = "./data/user_study_gaze.txt"
    f = open(test_file,"w")
    print f
    
    # true_rbf = autils.RbfComplexReward(obj_centers, obj_weights, abs_weights)
    print(true_rbf.obj_weights)
    print(obj_weights)
    test_rbfs = test_placements(true_rbf, num_test)
    mean_obj_wts, mean_abs_wts = birl.get_map_params()
    print "obj weights", mean_obj_wts
    print "abs weights", mean_abs_wts
    ave_loss, std_loss, max_loss = calc_test_reward_loss(test_rbfs, birl.get_map_params(), False)
    f.write("policy loss,{},{},{}\n".format(ave_loss, std_loss, max_loss))
    print "policy loss:", ave_loss, std_loss, max_loss

    ave_loss, std_loss, max_loss = calc_test_placement_loss(test_rbfs, birl.get_map_params(), False)
    f.write("placement loss,{},{},{}\n".format(ave_loss, std_loss, max_loss))
    print "placement loss:", ave_loss, std_loss, max_loss
    print "reward diff:", np.linalg.norm(true_rbf.obj_weights - mean_obj_wts), np.linalg.norm(true_rbf.abs_weights - mean_abs_wts)
    f.close()

    #create rbf for MAP reward found by BIRL in test configurations to see how it generalizes
    for i in range(3):
        #I'm giving it random object placements to see if it generalizes
        rand_obj_centers = np.random.random((num_objects,2))
        map_reward_rbf = autils.RbfComplexReward(rand_obj_centers, map_obj_wts, map_abs_wts)

        #visualize learned reward and optimal placement
        autils.visualize_reward(map_reward_rbf, title="test {} for MAP reward".format(i))
    # plt.show()