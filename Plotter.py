# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 10:49:44 2021

@author: Henke
"""
import matplotlib.pyplot as plt
accuracies = True
loss = True
bound = True

#--Load acc_history from files
if accuracies == True and bound == False:
    #filenames = ["0noise-10hidden", "0.1noise-10hidden-trunc", "0.1noise-10hidden-henrik"]
    #filenames = ["0.1noise-100hidden-xiaowen", "0.1noise-100hidden-henrik"]
    #filenames = ["0noise-10hidden-xiaowen", "1noise-10hidden-xiaowen", "1noise-10hidden-xiaowen-1rtx"]
    #filenames = ["0.1noise-10hidden-xiaowen-10rtx", "0.1noise-10hidden-henrik-10rtx"]
    #filenames = ["1noise-10hidden-henrik-0rtx-50budget", "1noise-10hidden-henrik-1rtx-50budget", "1noise-10hidden-henrik-3rtx-50budget", "1noise-10hidden-henrik-7rtx-50budget"]
    #filenames = ["1noise-10hidden-henrik-0rtx-100budget", "1noise-10hidden-henrik-1rtx-100budget", "1noise-10hidden-henrik-3rtx-100budget", "1noise-10hidden-henrik-7rtx-100budget", "1noise-10hidden-henrik-15rtx-100budget", "1noise-10hidden-henrik-31rtx-100budget"]
    #filenames = ["1noise-10hidden-henrik-0rtx-100budget", "1noise-10hidden-henrik-3rtx-100budget", "1noise-10hidden-henrik-31rtx-100budget"]
    #filenames = ["1noise-10hidden-henrik-0rtx-200budget", "1noise-10hidden-henrik-1rtx-200budget", "1noise-10hidden-henrik-3rtx-200budget", "1noise-10hidden-henrik-0.1growth-200budget", "1noise-10hidden-henrik-0.2growth-200budget"]
    #filenames = ["0.7071067811865475noise-10hidden-henrik-0rtx-200budget", "0.7071067811865475noise-10hidden-henrik-7rtx-200budget", "0.7071067811865475noise-10hidden-henrik-63rtx-200budget"]
    #filenames = ["1noise-10hidden-henrik-3rtx-200budget", "1noise-10hidden-henrik-3rtx-200budget-2"]
    #filenames = ["2noise-10hidden-henrik-0rtx-200budget", "2noise-10hidden-henrik-7rtx-200budget", "2noise-10hidden-henrik-15rtx-200budget", "2noise-10hidden-henrik-31rtx-200budget", "2noise-10hidden-henrik-0.1growth-200budget", "2noise-10hidden-henrik-0.2growth-200budget"]
    #filenames = ["1.5noise-10hidden-henrik-0rtx-200budget", "1.5noise-10hidden-henrik-3rtx-200budget", "1.5noise-10hidden-henrik-7rtx-200budget", "1.5noise-10hidden-henrik-15rtx-200budget", "1.5noise-10hidden-henrik-31rtx-200budget"]
    #filenames = ["0noise-10hidden-henrik-0rtx-50budget", "10noise-10hidden-henrik-0rtx-50budget", "10noise-10hidden-henrik-4rtx-50budget", "10noise-10hidden-henrik-1rtx-50budget"]
    #filenames = ["0noise-10hidden-henrik-0rtx-50budget", "1noise-10hidden-henrik-0rtx-50budget", "2noise-10hidden-henrik-0rtx-50budget", "10noise-10hidden-henrik-0rtx-50budget", "20noise-10hidden-henrik-0rtx-50budget", "30noise-10hidden-henrik-0rtx-50budget", "40noise-10hidden-henrik-0rtx-50budget"]
    #filenames = ["20noise-10hidden-henrik-0rtx-100budget", "20noise-10hidden-henrik-1rtx-100budget", "20noise-10hidden-henrik-3rtx-100budget"]
    #filenames = ["3.1622776601683795noise-10hidden-henrik-0rtx-300budget", "3.1622776601683795noise-10hidden-henrik-1rtx-300budget"]
    #filenames = ["3.1622776601683795noise-10hidden-henrik-0rtx-400budget", "3.1622776601683795noise-10hidden-henrik-1rtx-400budget"]
    #filenames = ["8.94427190999916noise-10hidden-henrik-0rtx-400budget", "8.94427190999916noise-10hidden-henrik-1rtx-400budget"]
    #filenames = ["8.94427190999916noise-10hidden-henrik-0rtx-400budget", "8.94427190999916noise-10hidden-henrik-1rtx-400budget", "8.94427190999916noise-10hidden-henrik-2rtx-400budget", "8.94427190999916noise-10hidden-henrik-3rtx-400budget", "8.94427190999916noise-10hidden-henrik-4rtx-400budget", "8.94427190999916noise-10hidden-henrik-5rtx-400budget"]
    #filenames = ["9noise-10hidden-henrik-0rtx-20budget_loss", "9noise-10hidden-henrik-3rtx-20budget_loss", "9noise-10hidden-henrik-7rtx-20budget_loss"]
    #filenames = ["3noise-10hidden-henrik-0rtx-400budget", "3noise-10hidden-henrik-1rtx-400budget", "3noise-10hidden-henrik-2rtx-400budget", "3noise-10hidden-henrik-3rtx-400budget", "3noise-10hidden-henrik-4rtx-400budget", "3noise-10hidden-henrik-5rtx-400budget", "3noise-10hidden-henrik-6rtx-400budget"]
    #filenames = ["9noise-10hidden-henrik-0rtx-200budget", "9noise-10hidden-henrik-1rtx-200budget", "9noise-10hidden-henrik-3rtx-200budget", "9noise-10hidden-henrik-7rtx-200budget"]
    #filenames = ["10.0noise-100hidden-henrik-0rtx-150budget", "10.0noise-100hidden-henrik-1rtx-150budget", "10.0noise-100hidden-henrik-3rtx-150budget", "10.0noise-100hidden-henrik-7rtx-150budget"]
    #filenames = ["10.0noise-100hidden-henrik-0rtx-200budget", "10.0noise-100hidden-henrik-2rtx-200budget", "10.0noise-100hidden-henrik-4rtx-200budget", "10.0noise-100hidden-henrik-6rtx-200budget", "10.0noise-100hidden-henrik-8rtx-200budget", "10.0noise-100hidden-henrik-10rtx-200budget"]
    #filenames = ["10.0noise-10hidden-henrik-0rtx-200budget-10numav", "10.0noise-10hidden-henrik-2rtx-200budget-10numav", "10.0noise-10hidden-henrik-4rtx-200budget-10numav", "10.0noise-10hidden-henrik-6rtx-200budget-10numav", "10.0noise-10hidden-henrik-8rtx-200budget-10numav"]
    filenames = ["3.1622776601683795noise-10hidden-henrik-0rtx-80budget-30numav", "3.1622776601683795noise-10hidden-henrik-1rtx-80budget-30numav", "3.1622776601683795noise-10hidden-henrik-3rtx-80budget-30numav", "3.1622776601683795noise-10hidden-henrik-7rtx-80budget-30numav"]
#--Load loss history for heuristic simulation
if loss == True or bound == True:
    #filenames = ["10.0noise-100hidden-henrik-0rtx-150budget_loss", "10.0noise-100hidden-henrik-1rtx-150budget_loss", "10.0noise-100hidden-henrik-3rtx-150budget_loss", "10.0noise-100hidden-henrik-7rtx-150budget_loss"]
    #filenames = ["10.0noise-100hidden-henrik-0rtx-200budget_loss", "10.0noise-100hidden-henrik-2rtx-200budget_loss", "10.0noise-100hidden-henrik-4rtx-200budget_loss", "10.0noise-100hidden-henrik-6rtx-200budget_loss", "10.0noise-100hidden-henrik-8rtx-200budget_loss", "10.0noise-100hidden-henrik-10rtx-200budget_loss"]
    #filenames = ["10.0noise-10hidden-henrik-0rtx-200budget-10numav_loss", "10.0noise-10hidden-henrik-2rtx-200budget-10numav_loss", "10.0noise-10hidden-henrik-4rtx-200budget-10numav_loss", "10.0noise-10hidden-henrik-6rtx-200budget-10numav_loss", "10.0noise-10hidden-henrik-8rtx-200budget-10numav_loss"]
    filenames = ["3.1622776601683795noise-10hidden-henrik-0rtx-80budget-30numav_loss", "3.1622776601683795noise-10hidden-henrik-1rtx-80budget-30numav_loss", "3.1622776601683795noise-10hidden-henrik-3rtx-80budget-30numav_loss", "3.1622776601683795noise-10hidden-henrik-7rtx-80budget-30numav_loss"]
#--Load norm_history from files
#filenames = ["1noise-10hidden-henrik-3rtx-101budget_norm", "1noise-10hidden-henrik-3rtx-101budget_error"]
#filenames = ["20noise-10hidden-henrik-1rtx-100budget_norm", "20noise-10hidden-henrik-1rtx-100budget_error", "20noise-10hidden-henrik-1rtx-100budget_rcv"]
#filenames = ["1noise-10hidden-henrik-9rtx-200budget_norm", "2noise-10hidden-henrik-0rtx-200budget_norm"]
#filenames = ["2noise-10hidden-henrik-0rtx-20budget_norm", "2noise-10hidden-henrik-0rtx-20budget_error"]
#filenames = ["4noise-10hidden-henrik-0rtx-20budget_norm", "4noise-10hidden-henrik-0rtx-20budget_error"]
#filenames = ["8noise-10hidden-henrik-0rtx-50budget_norm", "8noise-10hidden-henrik-0rtx-50budget_error"]
#filenames = ["10noise-10hidden-henrik-0rtx-50budget_norm", "10noise-10hidden-henrik-0rtx-50budget_error", "10noise-10hidden-henrik-0rtx-50budget_rcv"]
#filenames = ["0noise-10hidden-henrik-0rtx-50budget_norm", "0noise-10hidden-henrik-0rtx-50budget_error", "0noise-10hidden-henrik-0rtx-50budget_rcv"]

ymin= 0
ymax = 0
if accuracies == True:
    ymin = 0.6
    ymax = 0.92
else:
    ymin = 0.3
    ymax = 0.4
acc_histories = []

for filename in filenames:
    acc_history = []
    with open("./data/" + filename + ".txt", "r") as filehandle:
        for line in filehandle:
            acc_history.append(float(line))
    acc_histories.append(acc_history)
    
#Find max length
max_len = 1
for acc_history in acc_histories:
    if len(acc_history) > max_len:
        max_len = len(acc_history)
        
if bound == False:
    #--Plot acc_histories
    start = 0
    end = -1
    fixed_labels = True
    if fixed_labels == False:
        labels = filenames
    else:
        if accuracies == True or loss == True:
            labels = ["M=1", "M=2", "M=4", "M=8", "M=5", "M=6", "M=7", "M=8", "M=9", "M=10", "M=0.1growth", "M=0.2growth", "M=0.3growth"]
        else:
            labels = ["norm", "error", "rcv_norm"]
        #labels = ["sigma=0", "sigma=1", "sigma=2", "sigma=10", "sigma=20", "sigma=30", "sigma=40"]
    fig = plt.figure()
    i = 0
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for acc_history in acc_histories:
        plt.plot(range(len(acc_history)), acc_history, colors[i], label=labels[i])
        dashed_line_length = max_len-len(acc_history)
        plt.hlines(acc_history[len(acc_history)-1], len(acc_history)-1, max_len, colors[i], linestyles='dashed')
        plt.ylim((ymin,ymax))
        #plt.plot(range(dashed_line_length), [acc_history[len(acc_history)-1]]*dashed_line_length, colors[i] + "--", xmin=len(acc_history))
        i = i + 1
    plt.xlabel("Communication Round")
    if accuracies == True:
        plt.ylabel("Classification Accuracy")
    elif loss == True:
        plt.ylabel("FL Loss")
    plt.legend(loc=0)
    #compareplot_name = "200budget-1noise-comparison"
    compareplot_name = "100budget-unnormalized-norm-comparison"
    plt.savefig("./plots/"+compareplot_name+".png", format='png')

#--Plot bound
if bound == True:
    loss_histories = acc_histories
    bounds = []
    filename = "3.1622776601683795noise-10hidden-henrik-bound-80budget-30numav"
    with open("./data/" + filename + ".txt", "r") as filehandle:
        for line in filehandle:
            bounds.append(float(line))
    
    final_losses = []
    for loss_history in loss_histories:
        final_losses.append(loss_history[-1])
        
    #scaling
    factor = 10
    zero_point = final_losses[0]*factor
    for i in range(len(final_losses)):
        final_losses[i] = final_losses[i]*factor
        print(final_losses[i])
        final_losses[i] = final_losses[i] - zero_point + 0
        
    M_list = [1, 2, 4, 8]
        
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    plt.plot(M_list, bounds, colors[0], label="bound")
    plt.plot(M_list, final_losses, colors[1], label="loss")
    plt.xlabel("M")
    plt.ylabel("Final Loss")
    plt.legend()
        