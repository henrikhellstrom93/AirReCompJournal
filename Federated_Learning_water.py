# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 13:24:19 2021

@author: Henke
"""
import math
import HelperFunctions
import tensorflow as tf
import time
import numpy as np
from Power_Control import PowerControl
from tensorflow.keras.optimizers import SGD

#--Simulation settings--
static = True
if static == True:
    rtx_list = [0, 1, 3, 7, 15]
    #rtx_list = [0, 1]
else:
    growth = 0.2
    #These three variables doesn't do anything for dynamic retransmissions,
    #I should really make the program nicer than this, but not now.
    rtx_max = 0
    search_rtx = False
    rtx_fix = 0
uplink_budget = 90 # Uplink transmission constraint
normalize = True
num_devices = 10 # Number of devices
num_av = 50
num_hidden = 10
bs = 50 # Batch size for local training at devices
sigma_w = np.sqrt(10*num_devices) # Noise variance
#sigma_w = 1
beta = 0.05 # Learning rate
C_bar = uplink_budget #Cost budget
C_u = 1 #Cost of uplink transmission
C_t = 2 #Cost of local training
ep = 2 # Number of local epochs before communication round
filename_start = str(sigma_w) + "noise-" + str(num_hidden) + "hidden-henrik-"
if static == True:
    filename_end = "rtx-" + str(uplink_budget) + "budget-" + str(num_av) + "numav"
else:
    filename_end = str(growth) + "growth-" + str(uplink_budget) + "budget"
powercontrol = "henrik"
# powercontrol = "xiaowen"

#--Load iWater dataset--

water_data = HelperFunctions.loadWaterDataset("./database_water/")

#Shuffle dataset since we are not utilizing time
np.random.shuffle(water_data)

#Conductivity is considered the output
x = water_data[:,1:]
y = water_data[:,0]

print("std = ", np.std(x[:,0]))
quit()

#Normalize input data
for i in range(5):
    x[:,i] = x[:,i]/np.std(x[:,i])
    x[:,i] = x[:,i] - np.mean(x[:,i])

#Change problem into binary classification
y = (np.greater(y, 0)).astype(int)

#25000 train samples 5000 test samples
x_train = x[0:25000,:]
x_test = x[25000:,:]
y_train = y[0:25000]
y_test = y[25000:]

num_samples = len(x_train)

#Split dataset into shards
samples_per_shard = int(num_samples/num_devices)
x_train_shards = []
y_train_shards = []
for i in range(num_devices):
    x_train_shard = x_train[i*samples_per_shard:(i+1)*samples_per_shard]
    x_train_shards.append(x_train_shard)
    y_train_shard = y_train[i*samples_per_shard:(i+1)*samples_per_shard]
    y_train_shards.append(y_train_shard)

print("Dataset loaded.")

ac_loss_histories = []
av_mse_histories = []
norm_history = []
rcv_norm_history = []
comm_error_history = []
bounds = np.zeros((len(rtx_list), 1))

#pc is instantiated before loops to make sure all different M are compared with the same fading channel
pc = PowerControl(num_devices, sigma_w)

#--Calculate bound for heuristic--
K = num_devices
h = pc.h
i = 0
for rtx in rtx_list:
    pc.setRtx(rtx)
    M = rtx+1
    N = math.floor(C_bar/(C_t+M*C_u))
    eta = pc.henrikEta()
    pc.eta_h = eta
    b_h = pc.henrikB()
    p = np.square(np.abs(b_h))
    bound = K*np.sqrt(eta)/(2*N*beta*np.dot(np.transpose(np.sqrt(p)), np.abs(h)))
    bounds[i] = bound[0][0]
    i = i + 1
print("Bound = ", bounds)
        
for a in range(num_av):
    final_mses = []
    loss_histories = []
    mse_histories = []
    for rtx in rtx_list:
        M = rtx+1 # Number of uplink transmissions per round
        print("M=", M)
        print("h=", np.abs(pc.h))
        print("a=", a)
        num_rounds = math.floor(C_bar/(C_t+M*C_u)) # Number of communication rounds
        
        #--Set up DNN models--
        model_template = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(5,1)),
            tf.keras.layers.Dense(num_hidden, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        model_list = []
        
        #Create user device models and global model
        global_model = tf.keras.models.clone_model(model_template)
        for i in range(num_devices):
            model_list.append(tf.keras.models.clone_model(model_template))
        
        loss_fn = tf.keras.losses.MeanSquaredError()
        
        #Create optimizer
        opt = SGD(lr=beta, momentum=0, decay=0)
        
        #Compile all models to initiate weights
        for model in model_list:
            model.compile(optimizer=opt, loss=loss_fn, metrics=[tf.keras.metrics.MeanSquaredError()])
        
        #Broadcast global model to all user devices
        global_weights = global_model.get_weights()
        for model in model_list:
            model.set_weights(global_weights)
            
        #--Perform FL--
        pc.setStatic(static)
        if static == True:
            pc.setRtx(rtx)
        else:
            pc.setBudgetGrowth(uplink_budget, growth)
            
        loss_history = []
        mse_history = []
        for r in range(num_rounds):
            print("Communication round " + str(r+1) + "/" + str(num_rounds))
            pc.newRound()
            if static == False:
                if pc.getBudget() <= 0:
                    break
            start = time.time()
            #Train using local dataset
            for d in range(num_devices): #TODO: Parallelize
                model_list[d].fit(x_train_shards[d], y_train_shards[d], batch_size=bs, epochs=ep, verbose=0)
            
            training_time = time.time()-start
            
            #Calculate weight updates
            new_weights = []
            weight_updates = []
            for d in range(num_devices):
                new_weights.append(model_list[d].get_weights())
                weight_updates.append(new_weights[d]) #Just to initiate shape
                for l in range(len(weight_updates[d])):
                    weight_updates[d][l] = new_weights[d][l] - global_weights[l]
                    
            update_time = time.time()-training_time-start
            num_layers = len(weight_updates[0])
            
            if normalize == True:
                #Normalize before transmission
                update_vectors = HelperFunctions.vectorifyUpdates(weight_updates, False)
                std = np.zeros((num_devices, 1))
                for d in range(num_devices):
                    std[d] = np.std(np.abs(update_vectors[d]))
                    for l in range(num_layers):
                        weight_updates[d][l] = weight_updates[d][l]/std[d]
                #print("Parameter mean = ", np.mean(update_vectors))
                    
            #Calculate vector norms for plotting
            update_vectors = HelperFunctions.vectorifyUpdates(weight_updates, False)
            mean_update_vector = HelperFunctions.calculateMeanVector(update_vectors)
            mean_update_norm = np.linalg.norm(mean_update_vector)
            norm_history.append(mean_update_norm)
            
            #Power Control to get sum
            sum_update = weight_updates[0] #Just to initiate shape
            for l in range(num_layers):
                #Even layers contain weights
                if l%2 == 0:
                    layer_height = len(weight_updates[0][l])
                    layer_width = len(weight_updates[0][l][0])
                    #We get one row of the weight matrix at a time
                    for r in range(layer_height):
                        row_matrix = np.zeros((num_devices, layer_width))
                        for d in range(num_devices):
                            row = weight_updates[d][l][r]
                            row_matrix[d,:] = row
                        row_sum = pc.estLayer(row_matrix, powercontrol)
                        if r == 100:
                            #quit()
                            pass
                        sum_update[l][r] = row_sum[:].reshape(layer_width,)
                else:
                    layer_width = len(weight_updates[0][l])
                    row_matrix = np.zeros((num_devices, layer_width))
                    for d in range(num_devices):
                        row = weight_updates[d][l]
                        row_matrix[d,:] = row
                    row_sum = pc.estLayer(row_matrix, powercontrol)
                    sum_update[l] = row_sum[:].reshape(layer_width,)
            
            power_time = time.time()-update_time-training_time-start
                    
            #Average at server
            average_update = sum_update
            for l in range(num_layers):
                average_update[l] = average_update[l]/num_devices
            
            rcv_update_vector = HelperFunctions.vectorifyUpdates(average_update, True)
            comm_diff = mean_update_vector-rcv_update_vector
            comm_error = np.linalg.norm(comm_diff)
            comm_error_history.append(comm_error)
            rcv_norm = np.linalg.norm(rcv_update_vector)
            rcv_norm_history.append(rcv_norm)
            
            if normalize == True:
                #Denormalize using std measures from devices
                for l in range(num_layers):
                    mean_std = np.mean(std)
                    average_update[l] = average_update[l]*mean_std
            
            #Update global model
            new_global = global_weights #Just to initiate shape
            for l in range(len(new_global)):
                new_global[l] = global_weights[l] + average_update[l]
            #Broadcast global model to user devices
            for model in model_list:
                model.set_weights(new_global)
            global_weights = new_global
            #Logging
            loss_history.append(model_list[0].evaluate(x_test, y_test, verbose=0)[0])
            mse_history.append(model_list[0].evaluate(x_test, y_test, verbose=0)[1])
            print("Current test dataset MSE: ", mse_history[-1])
            print(str(int(training_time)) + " training seconds elapsed\n")
            print(str(int(update_time)) + " update seconds elapsed\n")
            print(str(int(power_time)) + " power control seconds elapsed\n")
            print(str(int(time.time()-start)) + " total seconds elapsed\n") 
            
        loss_histories.append(loss_history)
        mse_histories.append(mse_history)
        final_mses.append(mse_history[-1])
    if a == 0:
        av_loss_histories = loss_histories
        av_mse_histories = mse_histories
    else:
        for b in range(len(mse_histories)):
            for c in range(len(mse_histories[b])):
                av_loss_histories[b][c] = av_loss_histories[b][c] + loss_histories[b][c]
                av_mse_histories[b][c] = av_mse_histories[b][c] + mse_histories[b][c]

for a in range(len(mse_histories)):
    for b in range(len(mse_histories[a])):
        av_loss_histories[a][b] = av_loss_histories[a][b]/num_av
        av_mse_histories[a][b] = av_mse_histories[a][b]/num_av

rtx_index = 0
for mse_history in av_mse_histories:
    rtx = rtx_list[rtx_index]
    if static == True:
        filename = filename_start + str(rtx) + filename_end
    else:
        filename = filename_start + filename_end
    #--Plot MSE--
    import matplotlib.pyplot as plt
    start = 0
    end = -1
    fig = plt.figure()
    plt.plot(range(len(mse_history)), mse_history)
    plt.savefig("./plots/"+filename+".png", format='png')
    
    #--Store mse_history in file--
    with open("./data_water/"+filename+".txt", "w") as filehandle:
        for item in mse_history:
            filehandle.write("%s\n" % item)
            
    #--Store norm history in file--
    filename_norm = filename + "_norm"
    with open("./data_water/"+filename_norm+".txt", "w") as filehandle:
        for item in norm_history:
            filehandle.write("%s\n" % item)
            
    #--Store error history in file--
    filename_error = filename + "_error"
    with open("./data_water/"+filename_error+".txt", "w") as filehandle:
        for item in comm_error_history:
            filehandle.write("%s\n" % item)
            
    #--Store rcv norm history in file--
    filename_rcv = filename + "_rcv"
    with open("./data_water/"+filename_rcv+".txt", "w") as filehandle:
        for item in rcv_norm_history:
            filehandle.write("%s\n" % item)
    
    print("Stored results in file:", filename)
    rtx_index = rtx_index + 1
    
#--Store loss--
rtx_index = 0
for loss_history in av_loss_histories:
    rtx = rtx_list[rtx_index]
    if static == True:
        filename = filename_start + str(rtx) + filename_end
    else:
        filename = filename_start + filename_end
    filename_loss = filename + "_loss"
    with open("./data_water/"+filename_loss+".txt", "w") as filehandle:
        for item in loss_history:
            filehandle.write("%s\n" % item)
    rtx_index = rtx_index + 1
    
#--Store bound--
filename = filename_start + "bound-" + str(uplink_budget) + "budget-" + str(num_av) + "numav"
with open("./data_water/"+filename+".txt", "w") as filehandle:
    for item in bounds:
        filehandle.write("%s\n" % item[0])