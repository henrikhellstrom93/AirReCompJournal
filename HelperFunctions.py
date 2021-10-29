# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 12:13:18 2021

@author: Henke
"""
import numpy as np

def vectorifyUpdates(weight_updates, single_model):
    #If we got multiple neural networks
    if single_model == False:
        num_devices = len(weight_updates)
        num_layers = int(len(weight_updates[0])/2)
        num_weights_layer1 = len(weight_updates[0][0])*len(weight_updates[0][0][0])
        num_bias_layer1 = len(weight_updates[0][1])
        num_weights_layer2 = len(weight_updates[0][2])*len(weight_updates[0][2][0])
        num_bias_layer2 = len(weight_updates[0][3])
        vector_length = num_weights_layer1+num_bias_layer1+num_weights_layer2+num_bias_layer2
        ret_vectors = []
        for i in range(num_devices):
            ret_vector = np.zeros((vector_length,1))
            pointer = 0
            for j in range(num_layers*2):
                #If j%2 == 0, then we are looking at weights
                if j % 2 == 0:
                    weight_vector = np.concatenate(weight_updates[i][j])
                    num_weights = len(weight_vector)
                    ret_vector[pointer:pointer+num_weights] = weight_vector.reshape((num_weights,1))
                    pointer = pointer + num_weights
                #If j%2 == 1, then we are looking at biases
                elif j % 2 == 1:
                    bias_vector = weight_updates[i][j]
                    num_biases = len(bias_vector)
                    ret_vector[pointer:pointer+num_biases] = bias_vector.reshape((num_biases,1))
                    pointer = pointer + num_biases
            ret_vectors.append(ret_vector)
        return ret_vectors
    #If we only got one model
    else:
        num_layers = int(len(weight_updates)/2)
        num_weights_layer1 = len(weight_updates[0])*len(weight_updates[0][0])
        num_bias_layer1 = len(weight_updates[1])
        num_weights_layer2 = len(weight_updates[2])*len(weight_updates[2][0])
        num_bias_layer2 = len(weight_updates[3])
        vector_length = num_weights_layer1+num_bias_layer1+num_weights_layer2+num_bias_layer2
        ret_vector = np.zeros((vector_length,1))
        pointer = 0
        for j in range(num_layers*2):
            #If j%2 == 0, then we are looking at weights
            if j % 2 == 0:
                weight_vector = np.concatenate(weight_updates[j])
                num_weights = len(weight_vector)
                ret_vector[pointer:pointer+num_weights] = weight_vector.reshape((num_weights,1))
                pointer = pointer + num_weights
            #If j%2 == 1, then we are looking at biases
            elif j % 2 == 1:
                bias_vector = weight_updates[j]
                num_biases = len(bias_vector)
                ret_vector[pointer:pointer+num_biases] = bias_vector.reshape((num_biases,1))
                pointer = pointer + num_biases
        return ret_vector

def calculateMeanVector(vector_list):
    num_vectors = len(vector_list)
    ret_vector = np.zeros(vector_list[0].shape)
    for i in range(num_vectors):
        ret_vector = ret_vector + vector_list[i]
    ret_vector = ret_vector/num_vectors
    return ret_vector

#Input: path = path to folder with iWater dataset
#Output: returns a 30000x6 matrix where the rows contain:
#row 0: Conductivity
#row 1: oxygen saturation
#row 2: ph value
#row 3: redox
#row 4: salinity
#row 5: temperature
def loadWaterDataset(path):
    ret = np.zeros((30000, 6))
    filenames = ["iWater_node_01_CN_A.txt", "iWater_node_01_OS_D.txt", "iWater_node_01_PH_C.txt", "iWater_node_01_RX_C.txt", "iWater_node_01_SA_A.txt", "iWater_node_01_TC1_D.txt"]
    file_index = 0
    for filename in filenames:
        with open(path+filename, "r") as f:
            raw_line = f.read()
        
        date_and_tsv = raw_line.split()
        timestamps_and_values = date_and_tsv[1::2]
        values = np.zeros((30000,1))
        
        for i in range(30000):
            timestamp_and_value = timestamps_and_values[i]
            value = timestamp_and_value[9:]
            values[i] = value
            
        ret[:,file_index] = values.ravel()
        file_index = file_index + 1
        
    return ret