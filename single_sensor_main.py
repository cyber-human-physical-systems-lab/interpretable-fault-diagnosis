import os
import copy
import numpy as np
from utils import shuffle_data_single, load_data, init_formula, cal_robustness, gradient_decent
from itertools import combinations

dataset_path = os.path.join(os.getcwd(), 'datasets')
write_dir = os.path.join(os.getcwd(), 'outputs', 'single')
os.makedirs(write_dir, exist_ok=True)

## single sensor
sensor_idx = 0
data_name = 'data_SINGLE'
cur_data_path = os.path.join(dataset_path, f'{data_name}.mat')
multi_flag = 0

## shuffle data, split them as train & test, and save them as numpy arrays
shuffle_data_single(cur_data_path, dataset_path, data_name)
time, signal, label, normalize_param = load_data(dataset_path, data_name, multi_flag)

## initialize formula
init_f = init_formula(sensor_idx, time['train'][0], signal['train'])

if os.path.exists(os.path.join(write_dir, 'num.npy')):
    num = int(np.load(os.path.join(write_dir, 'num.npy')))
else:
    num = 0
write_sub_dir = os.path.join(write_dir, f'test_{num}')
os.makedirs(write_sub_dir, exist_ok=True)
np.save(os.path.join(write_dir, 'num.npy'), num + 1)

max_iter = 100
penalty_lambda = 1
max_del_count = 2
cur_formula = init_f
robustness_res = cal_robustness(time['test'], signal['test'], cur_formula, multi_flag)
robustness_res = robustness_res * label['test']
cur_accuracy = np.sum(robustness_res > 0.) / robustness_res.shape[0]
top_formula = copy.deepcopy(cur_formula)
top_accuracy = cur_accuracy
exist_formula_list = list(range(len(cur_formula)))
while len(exist_formula_list) > 0:
    write_path = os.path.join(write_sub_dir, f'formula_count_{len(exist_formula_list)}')
    os.makedirs(write_path, exist_ok=True)
    with open(os.path.join(write_path, 'info.txt'), 'a') as ftxt:
        ftxt.write(f'normalize_param: {normalize_param}\n')
        ftxt.write(f'exist_formula_list: {exist_formula_list}\n')
    top_formula, top_accuracy, top_iter = gradient_decent(label, time, signal, top_formula, penalty_lambda, len(exist_formula_list), max_iter, write_path, multi_flag)
    with open(os.path.join(write_sub_dir, 'info.txt'), 'a') as ftxt:
        ftxt.write(f'cur_formula_list: {exist_formula_list}\n')
        ftxt.write(f'cur_formula: {top_formula}\n')
        ftxt.write(f'cur_acc: {top_accuracy}\n')
        ftxt.write(f'iter: {top_iter}\n\n\n')
    if top_accuracy >= 0.9 * cur_accuracy:
        cur_accuracy = top_accuracy
        cur_formula = copy.deepcopy(top_formula)
        del_combo_list = []
        for k in range(1, max_del_count + 1):
            del_f = list(combinations(exist_formula_list, k))
            for dd in del_f:
                dd = list(dd)
                if len(exist_formula_list) - len(dd) > 0:
                    del_combo_list.append(dd)
        if len(del_combo_list) > 0:
            cost_list = []
            temp_formula_list = []
            for del_combo in del_combo_list:
                temp_formula = copy.deepcopy(top_formula)
                for e in del_combo:
                    temp_formula[e][1] = 0
                temp_formula_list.append(temp_formula)
                temp_robustness_res = cal_robustness(time['train'], signal['train'], temp_formula, multi_flag)
                temp_robustness_res = temp_robustness_res * label['train']
                temp_cost = np.sum(np.exp(-1 * temp_robustness_res)) + penalty_lambda * (len(exist_formula_list) - len(del_combo))
                cost_list.append(temp_cost)
            cost_min_index = cost_list.index(np.min(cost_list))
            top_formula = copy.deepcopy(temp_formula_list[cost_min_index])
            for ii in del_combo_list[cost_min_index]:
                exist_formula_list.remove(ii)
        else:
            break
    else:
        break