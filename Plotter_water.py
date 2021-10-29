# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 15:17:10 2021

@author: Henri
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 10:49:44 2021

@author: Henke
"""
import matplotlib.pyplot as plt
MSEs = True
loss = False
bound = False

#--Load MSE_history from files
if MSEs == True and bound == False:
    filenames = ["10.0noise-10hidden-henrik-0rtx-90budget-50numav", "10.0noise-10hidden-henrik-1rtx-90budget-50numav", "10.0noise-10hidden-henrik-3rtx-90budget-50numav", "10.0noise-10hidden-henrik-7rtx-90budget-50numav", "10.0noise-10hidden-henrik-15rtx-90budget-50numav"]
#--Load loss history for heuristic simulation
if loss == True or bound == True:
    filenames = ["10.0noise-10hidden-henrik-0rtx-90budget-50numav_loss", "10.0noise-10hidden-henrik-1rtx-90budget-50numav_loss", "10.0noise-10hidden-henrik-3rtx-90budget-50numav_loss", "10.0noise-10hidden-henrik-7rtx-90budget-50numav_loss", "10.0noise-10hidden-henrik-15rtx-90budget-50numav_loss"]
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
if MSEs == True:
    ymin = 0.0
    ymax = 0.2
else:
    ymin = 0.3
    ymax = 0.4
MSE_histories = []

for filename in filenames:
    MSE_history = []
    with open("./data_water/" + filename + ".txt", "r") as filehandle:
        for line in filehandle:
            MSE_history.append(float(line))
    MSE_histories.append(MSE_history)
    
#Find max length
max_len = 1
for MSE_history in MSE_histories:
    if len(MSE_history) > max_len:
        max_len = len(MSE_history)
        
if bound == False:
    #--Plot MSE_histories
    start = 0
    end = -1
    fixed_labels = True
    if fixed_labels == False:
        labels = filenames
    else:
        if MSEs == True or loss == True:
            labels = ["M=1", "M=2", "M=4", "M=8", "M=16", "M=6", "M=7", "M=8", "M=9", "M=10", "M=0.1growth", "M=0.2growth", "M=0.3growth"]
        else:
            labels = ["norm", "error", "rcv_norm"]
        #labels = ["sigma=0", "sigma=1", "sigma=2", "sigma=10", "sigma=20", "sigma=30", "sigma=40"]
    fig = plt.figure()
    i = 0
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for MSE_history in MSE_histories:
        plt.plot(range(len(MSE_history)), MSE_history, colors[i], label=labels[i])
        dashed_line_length = max_len-len(MSE_history)
        plt.hlines(MSE_history[len(MSE_history)-1], len(MSE_history)-1, max_len, colors[i], linestyles='dashed')
        plt.ylim((ymin,ymax))
        i = i + 1
    plt.xlabel("Communication Round")
    if MSEs == True:
        plt.ylabel("Mean Squared Error")
    elif loss == True:
        plt.ylabel("FL Loss")
    plt.legend(loc=0)
    #compareplot_name = "200budget-1noise-comparison"
    compareplot_name = "100budget-unnormalized-norm-comparison"
    plt.savefig("./plots/"+compareplot_name+".png", format='png')
    #Final losses
    print("Final losses:")
    M = 1
    for MSE_history in MSE_histories:
        print("M = ", M, ": ", MSE_history[-1])
        M = M * 2

#--Plot bound
if bound == True:
    loss_histories = MSE_histories
    bounds = []
    filename = "10.0noise-10hidden-henrik-bound-90budget-50numav"
    with open("./data_water/" + filename + ".txt", "r") as filehandle:
        for line in filehandle:
            bounds.append(float(line))
    
    final_losses = []
    for loss_history in loss_histories:
        final_losses.append(loss_history[-1])
        
    #scaling
    factor = 150
    zero_point = final_losses[0]*factor
    for i in range(len(final_losses)):
        final_losses[i] = final_losses[i]*factor
        print(final_losses[i])
        final_losses[i] = final_losses[i] - zero_point + 3.5
        
    M_list = [1, 2, 4, 8, 16]
        
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    plt.plot(M_list, bounds, colors[0], label="bound")
    plt.plot(M_list, final_losses, colors[1], label="loss")
    plt.xlabel("M")
    plt.ylabel("Final Loss")
    plt.legend()
        