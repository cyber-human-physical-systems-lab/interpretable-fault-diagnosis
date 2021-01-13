# Interpretable Fault Diagnosis for Cyber-Physical Systems: A Learning Perspective

## Introduction
This code package implements the interpretable fault diagnosis algorithm
from the paper "Interpretable Fault Diagnosis for Cyber-Physical Systems: A Learning Perspective".
This project introduces a learning-based method to enable CPSs to explain their faults to their human users in ahuman-friendly manner by learning an Signal Temporal Logic formula from the time-series datasets.
A case study with a spacecraft electrical power system is used to demonstratethe power of our method.
The dataset "data_ADAPT.mat" is generated on the VirtualADAPT simulink model developed by NASA. (https://github.com/nasa/VirtualADAPT)

## Environment
    python3

## Package Pre-requisites
    numpy
    pandas
    scipy
    itertools
    matplotlib

## Program Running
    Run single_sensor_main.py for learining an STL formula for a single sensor.
    Run multi_sensor_main.py for learining an STL formula for multple sensors.


    ./datasets/
        ------including two test datasets: "data_SINGLE.mat" for single-sensor data (artificially generated);
                                           "data_ADAPT.mat" for multi-sensor data.

    ./single_sensor_main.py
        ------code for running on single-sensor data

    ./multi_sensor_main.py
        ------code for running on multi-sensor data

## Output
    The outputs of this project includes the learned formula, the accuracy of each iteration
    and the cost of each iteration.
    For example, the learned formula is 
    ['1', '1', 'alw', '<', '53.76', '96.15', '1.34'], 
    ['3', '1', 'env', '<', '69.59', '113.43', '1.48'], 
    ['4', '1', 'env', '>', '80.37', '120.21', '1.30'].
    For each atom formula: the first value is the serial number of the sensor, 
    the second value "1/0" means if we keep this atom or not, the third value is "always/eventually", 
    the 5 and 6 values are the time patameters and the last value is the space parameter.

