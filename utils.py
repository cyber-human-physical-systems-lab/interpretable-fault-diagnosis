import os
import csv
import pandas as pd
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import random
import math


def judge(alist): 
    flag = 0
    for i in range(0, len(alist) - 1):
        if alist[i+1] > alist[i]:
            flag = 1
    if flag == 0:
        return True
    else:
        return False


def draw_signal(label, time, signal, write_path):
    plt.title('Sensor signal')
    for i in range(label.shape[0]):
        ll = int(label[i][0])
        tt = time[i][0].tolist()
        ss = signal[i][0].tolist()
        if ll == 1:
            color = 'green'
        else:
            color = 'red'
        plt.plot(tt, ss, color=color, label=str(ll))
    plt.xlabel('time')
    plt.ylabel('signal')
    plt.savefig(os.path.join(write_path, 'signal.png'))


def draw_res(write_path, cost_list, accuracy_list, cost_list_test, accuracy_list_test):
    x = range(0, len(cost_list))
    plt.figure()
    plt.title('cost')
    plt.plot(x, cost_list)
    plt.xlabel('iter')
    plt.ylabel('cost')
    plt.savefig(os.path.join(write_path, 'cost.png'))
    plt.close()

    plt.figure()
    plt.title('accuracy')
    plt.plot(x, accuracy_list)
    plt.xlabel('iter')
    plt.ylabel('accuracy')
    plt.savefig(os.path.join(write_path, 'accuracy.png'))
    plt.close()

    x = range(0, len(cost_list_test))
    plt.figure()
    plt.title('cost_test')
    plt.plot(x, cost_list_test)
    plt.xlabel('iter')
    plt.ylabel('cost')
    plt.savefig(os.path.join(write_path, 'cost_test.png'))
    plt.close()

    plt.figure()
    plt.title('accuracy_test')
    plt.plot(x, accuracy_list_test)
    plt.xlabel('iter')
    plt.ylabel('accuracy')
    plt.savefig(os.path.join(write_path, 'accuracy_test.png'))
    plt.close()


## shuffle data of single sensor, split as train & test, and save as numpy arrays
def shuffle_data_single(data_path, dataset_path, data_name):
    write_dir = os.path.join(dataset_path, data_name)
    org_numpy_dir = os.path.join(write_dir, 'org_numpy')
    if not os.path.exists(os.path.join(write_dir, 'org_data.csv')):
        os.makedirs(write_dir, exist_ok=True)
        os.makedirs(org_numpy_dir, exist_ok=True)
        row = ['index', 'time_shape', 'time_path', 'signal_shape', 'signal_path', 'label']
        with open(os.path.join(write_dir, 'org_data.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        cur_data = sio.loadmat(data_path)
        label = cur_data['label']
        trajs = cur_data['trajs']
        time = []
        signal = []
        data_count = trajs.shape[0]
        for i in range(trajs.shape[0]):
            time_shape = trajs[i]['time'][0].shape
            signal_shape = trajs[i]['X'][0].shape
            time.append(trajs[i]['time'][0])
            signal.append(trajs[i]['X'][0])
            time_np = trajs[i]['time'][0]
            signal_np = trajs[i]['X'][0]
            time_path = os.path.join(org_numpy_dir, f'time_{i}.npy')
            signal_path = os.path.join(org_numpy_dir, f'signal_{i}.npy')
            np.save(time_path, time_np)
            np.save(signal_path, signal_np)
            row = [i]
            row.append(time_shape)
            row.append(f'time_{i}.npy')
            row.append(signal_shape)
            row.append(f'signal_{i}.npy')
            row.append(int(label[i]))
            with open(os.path.join(write_dir, 'org_data.csv'), 'a') as f:
                writer = csv.writer(f)
                writer.writerow(row)
        x, y = time_shape
        time = np.array(time).reshape((len(time), x, y))
        signal = np.array(signal).reshape((len(signal), x, y))
        np.save(os.path.join(write_dir, 'org_time.npy'), time)
        np.save(os.path.join(write_dir, 'org_signal.npy'), signal)
        np.save(os.path.join(write_dir, 'org_label.npy'), label)

        df = pd.read_csv(os.path.join(write_dir, 'org_data.csv'))
        df = df.sample(frac=1)
        df.to_csv(os.path.join(os.path.join(write_dir, 'shuffled_data.csv')), index=0)
        shuffled_train_time = []
        shuffled_train_signal = []
        shuffled_train_label = []
        shuffled_test_time = []
        shuffled_test_signal = []
        shuffled_test_label = []
        with open(os.path.join(os.path.join(write_dir, 'shuffled_data.csv')), 'r') as f:
            info = csv.reader(f)
            for line in info:
                if info.line_num == 1:
                    line.append('category')
                else:
                    shuffled_time = np.load(os.path.join(org_numpy_dir, line[2]))
                    shuffled_signal = np.load(os.path.join(org_numpy_dir, line[4]))
                    shuffled_label = int(line[5])
                    line_num_count = int(data_count * 0.8) + 1
                    if info.line_num >= 2 and info.line_num <= line_num_count:
                        line.append('train')
                        shuffled_train_time.append(shuffled_time)
                        shuffled_train_signal.append(shuffled_signal)
                        shuffled_train_label.append(shuffled_label)
                    else:
                        line.append('test')
                        shuffled_test_time.append(shuffled_time)
                        shuffled_test_signal.append(shuffled_signal)
                        shuffled_test_label.append(shuffled_label)
                with open(os.path.join(os.path.join(write_dir, 'shuffled_data_division.csv')), 'a') as fcsv:
                    writer = csv.writer(fcsv)
                    writer.writerow(line)
        shuffled_train_time = np.array(shuffled_train_time).reshape((len(shuffled_train_time), x, y))
        shuffled_train_signal = np.array(shuffled_train_signal).reshape((len(shuffled_train_signal), x, y))
        shuffled_train_label = np.array(shuffled_train_label).reshape((len(shuffled_train_label), 1))
        np.save(os.path.join(write_dir, 'train_time.npy'), shuffled_train_time)
        np.save(os.path.join(write_dir, 'train_signal.npy'), shuffled_train_signal)
        np.save(os.path.join(write_dir, 'train_label.npy'), shuffled_train_label)

        shuffled_test_time = np.array(shuffled_test_time).reshape((len(shuffled_test_time), x, y))
        shuffled_test_signal = np.array(shuffled_test_signal).reshape((len(shuffled_test_signal), x, y))
        shuffled_test_label = np.array(shuffled_test_label).reshape((len(shuffled_test_label), 1))
        np.save(os.path.join(write_dir, 'test_time.npy'), shuffled_test_time)
        np.save(os.path.join(write_dir, 'test_signal.npy'), shuffled_test_signal)
        np.save(os.path.join(write_dir, 'test_label.npy'), shuffled_test_label)


def shuffle_data_multi(data_path, dataset_path, data_name):
    write_dir = os.path.join(dataset_path, data_name)
    org_numpy_dir = os.path.join(write_dir, 'org_numpy')
    if not os.path.exists(os.path.join(write_dir, 'org_data.csv')):
        os.makedirs(write_dir, exist_ok=True)
        os.makedirs(org_numpy_dir, exist_ok=True)
        row = ['index', 'time_shape', 'time_path', 'signal_shape', 'signal_path', 'label']
        with open(os.path.join(write_dir, 'org_data.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        cur_data = sio.loadmat(data_path)['data_ADAPT']
        data_count = cur_data.shape[1]
        total_time = []
        total_signal = []
        total_label = []
        ## sensor_index = 4
        ## plt.title(f'Sensor #{sensor_index} Signal')
        for i in range(cur_data.shape[1]):
            time = np.transpose(cur_data[:, i][0][0])
            read_signal = cur_data[:, i][0][1][0][0]
            label = int(cur_data[:, i][0][2])
            multi_sensor_signal = []
            ## tt = time[0].tolist()
            for signal in read_signal:
                signal = np.transpose(signal)
                multi_sensor_signal.append(signal)
            multi_sensor_signal = np.array(multi_sensor_signal).reshape((len(multi_sensor_signal), signal.shape[1]))
            time_path = os.path.join(org_numpy_dir, f'time_{i}.npy')
            signal_path = os.path.join(org_numpy_dir, f'signal_{i}.npy')
            np.save(time_path, time)
            np.save(signal_path, multi_sensor_signal)
            total_time.append(time)
            total_signal.append(multi_sensor_signal)
            total_label.append(label)
            row = [i]
            row.append(time.shape)
            row.append(f'time_{i}.npy')
            row.append(multi_sensor_signal.shape)
            row.append(f'signal_{i}.npy')
            row.append(label)
            with open(os.path.join(write_dir, 'org_data.csv'), 'a') as f:
                writer = csv.writer(f)
                writer.writerow(row)
        ##     ss = multi_sensor_signal[sensor_index, :].tolist()
        ##     if label == 1:
        ##         color = 'green'
        ##     else:
        ##         color = 'red'
        ##     plt.plot(tt, ss, color=color, label=str(label))
        ## plt.xlabel('time')
        ## plt.ylabel('signal')
        ## plt.savefig(os.path.join(write_dir, f'sensor{sensor_index}_signal.png'))
        total_time = np.array(total_time).reshape((len(total_time), time.shape[0], time.shape[1]))
        total_signal = np.array(total_signal).reshape((len(total_signal), multi_sensor_signal.shape[0], multi_sensor_signal.shape[1]))
        total_label = np.array(total_label).reshape((len(total_label), 1))
        np.save(os.path.join(write_dir, 'org_time.npy'), total_time)
        np.save(os.path.join(write_dir, 'org_signal.npy'), total_signal)
        np.save(os.path.join(write_dir, 'org_label.npy'), total_label)
        df = pd.read_csv(os.path.join(write_dir, 'org_data.csv'))
        df = df.sample(frac=1)
        df.to_csv(os.path.join(os.path.join(write_dir, 'shuffled_data.csv')), index=0)
        shuffled_train_time = []
        shuffled_train_signal = []
        shuffled_train_label = []
        shuffled_test_time = []
        shuffled_test_signal = []
        shuffled_test_label = []
        with open(os.path.join(os.path.join(write_dir, 'shuffled_data.csv')), 'r') as f:
            info = csv.reader(f)
            for line in info:
                if info.line_num == 1:
                    line.append('category')
                else:
                    shuffled_time = np.load(os.path.join(org_numpy_dir, line[2]))
                    shuffled_signal = np.load(os.path.join(org_numpy_dir, line[4]))
                    shuffled_label = int(line[5])
                    line_num_count = int(data_count * 0.8) + 1
                    if info.line_num >= 2 and info.line_num <= line_num_count:
                        line.append('train')
                        shuffled_train_time.append(shuffled_time)
                        shuffled_train_signal.append(shuffled_signal)
                        shuffled_train_label.append(shuffled_label)
                    else:
                        line.append('test')
                        shuffled_test_time.append(shuffled_time)
                        shuffled_test_signal.append(shuffled_signal)
                        shuffled_test_label.append(shuffled_label)
                with open(os.path.join(os.path.join(write_dir, 'shuffled_data_division.csv')), 'a') as fcsv:
                    writer = csv.writer(fcsv)
                    writer.writerow(line)
        shuffled_train_time = np.array(shuffled_train_time).reshape((len(shuffled_train_time), shuffled_time.shape[0], shuffled_time.shape[1]))
        shuffled_train_signal = np.array(shuffled_train_signal).reshape((len(shuffled_train_signal), shuffled_signal.shape[0], shuffled_signal.shape[1]))
        shuffled_train_label = np.array(shuffled_train_label).reshape((len(shuffled_train_label), 1))
        np.save(os.path.join(write_dir, 'train_time.npy'), shuffled_train_time)
        np.save(os.path.join(write_dir, 'train_signal.npy'), shuffled_train_signal)
        np.save(os.path.join(write_dir, 'train_label.npy'), shuffled_train_label)

        shuffled_test_time = np.array(shuffled_test_time).reshape((len(shuffled_test_time), shuffled_time.shape[0], shuffled_time.shape[1]))
        shuffled_test_signal = np.array(shuffled_test_signal).reshape((len(shuffled_test_signal), shuffled_signal.shape[0], shuffled_signal.shape[1]))
        shuffled_test_label = np.array(shuffled_test_label).reshape((len(shuffled_test_label), 1))
        np.save(os.path.join(write_dir, 'test_time.npy'), shuffled_test_time)
        np.save(os.path.join(write_dir, 'test_signal.npy'), shuffled_test_signal)
        np.save(os.path.join(write_dir, 'test_label.npy'), shuffled_test_label)
        print(shuffled_train_time.shape, shuffled_train_signal.shape, shuffled_train_label.shape)
        print(shuffled_test_time.shape, shuffled_test_signal.shape, shuffled_test_label.shape)


## load data
def load_data(dataset_path, data_name, multi_flag):
    train_time = np.load(os.path.join(dataset_path, data_name, 'train_time.npy'))
    train_signal = np.load(os.path.join(dataset_path, data_name, 'train_signal.npy'))
    train_label = np.load(os.path.join(dataset_path, data_name, 'train_label.npy'))
    test_time = np.load(os.path.join(dataset_path, data_name, 'test_time.npy'))
    test_signal = np.load(os.path.join(dataset_path, data_name, 'test_signal.npy'))
    test_label = np.load(os.path.join(dataset_path, data_name, 'test_label.npy'))

    if multi_flag == 0:
        normalize_param = np.max(train_signal) - np.min(train_signal)
        train_signal = train_signal / normalize_param
        test_signal = test_signal / normalize_param
    elif multi_flag == 1:
        normalize_param = []
        for sensor_idx in range(train_signal.shape[1]):
            sensor_train_signal = train_signal[:, sensor_idx, :]
            sensor_normalize_param = np.max(sensor_train_signal) - np.min(sensor_train_signal)
            normalize_param.append(sensor_normalize_param)
            train_signal[:, sensor_idx, :] /= sensor_normalize_param
            test_signal[:, sensor_idx, :] /= sensor_normalize_param
    else:
        print(f'WRONG! The value of multi_flag is {multi_flag} which shoule be 0(single-sensor) or 1(multi-sensor')
   
    time = {}
    signal = {}
    label = {}
    time['train'] = train_time
    time['test'] = test_time
    signal['train'] = train_signal
    signal['test'] = test_signal
    label['train'] = train_label
    label['test'] = test_label
    return time, signal, label, normalize_param


def init_formula(sensor_idx, time, signal):
    t_min = np.min(time)
    t_max = np.max(time)
    t_gap = (t_max - t_min) / 5
    tLeft_min = t_min + t_gap
    tLeft_max = t_min + 3 * t_gap
    sig_mean = np.mean(signal)
    sig_std = np.std(signal)
    ## print(t_min, t_max)
    ## print(sig_mean, sig_std)
    formula_list = []
    for i in range(4):
        tLeft = random.uniform(tLeft_min, tLeft_max)
        tRight_min = tLeft + t_gap
        tRight_max = min(tLeft_max, t_max - t_gap)
        tRight = random.uniform(tRight_min, tRight_max)
        sig_param = random.uniform(sig_mean - sig_std, sig_mean + sig_std)
        formula = [sensor_idx, 1]
        if i == 0:
            formula.append('alw')
            formula.append('>')
        elif i == 1:
            formula.append('alw')
            formula.append('<')
        elif i == 2:
            formula.append('env')
            formula.append('>')
        else:
            formula.append('env')
            formula.append('<')
        formula.append(tLeft)
        formula.append(tRight)
        formula.append(sig_param)
        formula_list.append(formula)
    return formula_list


def update_formula(cur_formula_list, new_params):
    new_formula_list = []
    new_params = new_params.tolist()
    for i in range(len(cur_formula_list)):
        cur_formula = cur_formula_list[i]
        new_formula = cur_formula[:4]
        new_formula.extend(new_params[i*3:i*3+3])
        new_formula_list.append(new_formula)
    return new_formula_list


def read_signal(input_signal, input_time, time_left, time_right):
    for i in range(input_time.shape[1]):
        if input_time[0, i] >= time_left:
            break
    for j in range(input_time.shape[1]):
        if input_time[0, j] >= time_right:
            j += 1
            break
    assert i < (j - 1), f'incorrect time range. time_left: {time_left}; time_right: {time_right}. Generated left index: {i}; right index: {j-1}'
    output_time = input_time[:, i:j]
    output_signal = input_signal[:, i:j]
    return output_time, output_signal


def cal_robustness_single(time, signal, formula_list, multi_flag):
    nu = 1
    ## exp_threshold = np.log(np.finfo(np.float64).max)
    robustness_list = []
    formula_robustness_list = []
    for i in range(time.shape[0]):
        cur_time = time[i]
        cur_signal = signal[i]
        robustness_seq = []
        for formula in formula_list:
            flag = int(formula[1])
            if flag == 1:
                cal_time, cal_signal = read_signal(input_signal=cur_signal, input_time=cur_time, time_left=float(formula[4]), time_right=float(formula[5]))
                cal_signal = cal_signal.astype(np.longdouble)
                if formula[2] == 'alw':
                    if formula[3] == '>':
                        rho_i = cal_signal - float(formula[6])
                    elif formula[3] == '<':
                        rho_i = float(formula[6]) - cal_signal
                    rho_min = np.min(rho_i)
                    rho_i_tuta = (rho_i - rho_min) / rho_min
                    if rho_min < 0.:
                        cur_robustness = (np.sum(rho_min * np.exp(rho_i_tuta) * np.exp(nu * rho_i_tuta))) / (np.sum(np.exp(nu * rho_i_tuta)))
                    elif rho_min > 0.:

                        cur_robustness = (np.sum(rho_i * np.exp(-1 * nu * rho_i_tuta))) / (np.sum(np.exp(-1 * nu * rho_i_tuta)))
                    else:
                        cur_robustness = np.sum([0.])
                else:
                    if formula[3] == '>':
                        rho_i = cal_signal - float(formula[6])
                    elif formula[3] == '<':
                        rho_i = float(formula[6]) - cal_signal
                    rho_max = np.max(rho_i)
                    rho_i_tuta = (rho_i - rho_max) / rho_max
                    if rho_max < 0.:
                        cur_robustness = (np.sum(rho_i * np.exp(-1 * nu * rho_i_tuta))) / (np.sum(np.exp(-1 * nu * rho_i_tuta))) 
                    elif rho_max > 0.:
                        cur_robustness = (np.sum(rho_max * np.exp(rho_i_tuta) * np.exp(nu * rho_i_tuta))) / (np.sum((np.exp(nu * rho_i_tuta)))) 
                    else:
                        cur_robustness = np.sum([0.])
                robustness_seq.append(cur_robustness)
            else:
                continue
        ## conjuction of the robustness of all selected formulas
        assert len(robustness_seq) > 0, 'incorrect robustness seq length. Please check the flag of formula_list'
        robustness_seq = np.array(robustness_seq)
        formula_robustness_list.append(robustness_seq)
        rho_formula_min = np.min(robustness_seq)
        rho_formula_i_tuta = (robustness_seq - rho_formula_min) / rho_formula_min
        if rho_formula_min < 0.:
            cur_formula_robustness = (np.sum(rho_formula_min * np.exp(rho_formula_i_tuta) * np.exp(nu * rho_formula_i_tuta))) / (np.sum(np.exp(nu * rho_formula_i_tuta)))
        elif rho_formula_min > 0.:
            cur_formula_robustness = (np.sum(robustness_seq * np.exp(-1 * nu * rho_formula_i_tuta))) / (np.sum(np.exp(-1 * nu * rho_formula_i_tuta)))
        else:
            cur_formula_robustness = np.sum([0.])
        robustness_list.append(cur_formula_robustness)
    robustness_np = np.expand_dims(np.array(robustness_list), axis=1)
    formula_robustness_list = np.array(formula_robustness_list)
    if multi_flag == 0:
        return robustness_np
    else:
        return formula_robustness_list
        # return robustness_np


def cal_robustness(time, total_signal, total_formula, multi_flag):
    if multi_flag == 1:
        nu = 1
        robustness_total_tmp = []
        for sensor_idx in range(total_signal.shape[1]):
            signal = np.expand_dims(total_signal[:, sensor_idx, :], axis=1)
            formula = total_formula[sensor_idx * 4 : (sensor_idx + 1) * 4]
            total_f = 0
            for f in formula:
                total_f += int(f[1])
            if total_f != 0:
                robustness_single = cal_robustness_single(time, signal, formula, multi_flag)
                robustness_total_tmp.append(robustness_single)
        assert len(robustness_total_tmp) > 0, 'Incorrect robustness seq length. It seems that there are no formula in the list'
        for i in range(len(robustness_total_tmp)):
            if i == 0:
                robustness_total = robustness_total_tmp[i]
            else:
                robustness_total = np.concatenate((robustness_total, robustness_total_tmp[i]), axis=1)
        r_total = []
        for i in range(robustness_total.shape[0]):
            r = robustness_total[i]
            rho_min = np.min(r)
            rho_i_tuta = (r - rho_min) / rho_min
            if rho_min < 0.:
                cur_r = (np.sum(rho_min * np.exp(rho_i_tuta) * np.exp(nu * rho_i_tuta))) / (np.sum(np.exp(nu * rho_i_tuta)))
            elif rho_min > 0.:
                cur_r = (np.sum(r * np.exp(-1 * nu * rho_i_tuta))) / (np.sum(np.exp(-1 * nu * rho_i_tuta)))
            else:
                cur_r = np.sum([0.])
            r_total.append(cur_r)
        r_total = np.expand_dims(np.array(r_total), axis=1)
    elif multi_flag == 0:
        r_total = cal_robustness_single(time, total_signal, total_formula, multi_flag)
    else:
        print(f'WRONG! The value of multi_flag is {multi_flag} which shoule be 0(single-sensor) or 1(multi-sensor')
    return r_total


def gradient_decent(label, time, signal, init_formula_list, penalty_lambda, penalty_count, max_iter, write_path, multi_flag):
    ## initialize
    cur_formula_list = init_formula_list

    ## calculate initial cost and accuracy
    robustness_res = cal_robustness(time['train'], signal['train'], cur_formula_list, multi_flag)
    robustness_res = robustness_res * label['train']
    cost = np.sum(np.exp(-1 * robustness_res)) + penalty_lambda * penalty_count
    accuracy = np.sum(robustness_res > 0.) / robustness_res.shape[0]

    robustness_res_test = cal_robustness(time['test'], signal['test'], cur_formula_list, multi_flag)
    robustness_res_test = robustness_res_test * label['test']
    cost_test = np.sum(np.exp(-1 * robustness_res_test)) + penalty_lambda * penalty_count
    accuracy_test = np.sum(robustness_res_test > 0.) / robustness_res_test.shape[0]

    cost_list = []
    accuracy_list = []
    cost_list.append(cost)
    accuracy_list.append(accuracy)
    cost_list_test = []
    accuracy_list_test = []
    cost_list_test.append(cost_test)
    accuracy_list_test.append(accuracy_test)
    print(f'Train\tinitial\tcost: {cost}\taccuracy: {accuracy}')
    print(f'Test\tinitial\taccuracy: {accuracy_test}')
    with open(os.path.join(write_path, f'info.txt'), 'a') as ftxt:
        ftxt.write(f'Train\tinitial\tcost: {cost}\taccuracy: {accuracy}\n')
        ftxt.write(f'Test\tinitial\taccuracy: {accuracy_test}\n')

    ## read current params that need to be updated
    params = []
    flags = []
    for cur_formula in cur_formula_list:
        flags.append(int(cur_formula[1]))
        for p in cur_formula[4:]:
            params.append(float(p))
    params = np.array(params)
    dim = params.shape

    ## params for gradient descent
    if multi_flag == 0: ## for single sensors
        h_t = 0.1
        h_s = 0.01
        alpha = 0.002  ## learning rate
    elif multi_flag == 1: ## for multiple sensors
        h_t = 1
        h_s = 0.01
        alpha = 0.001  ## learning rate
    else:
        print(f'WRONG! The value of multi_flag is {multi_flag} which shoule be 0(single-sensor) or 1(multi-sensor')

    acc_threshold = None  ## accuracy threshold for early stop
    momentum = 0.1
    b1 = 0.9  ## default
    b2 = 0.999  ## default
    e = 0.00000001  ## default
    mt = np.zeros(dim) ## initialize 1st moment vector
    vt = np.zeros(dim) ## initialize 1st moment vector
    stop_count = 5
    stop_accuracy_list = []
    write_path_iter = os.path.join(write_path, 'iter_init')
    os.makedirs(write_path_iter, exist_ok=True)
    np.save(os.path.join(write_path_iter, 'formula.npy'), cur_formula_list)
    np.save(os.path.join(write_path_iter, 'cost.npy'), cost_list)
    np.save(os.path.join(write_path_iter, 'accuracy.npy'), accuracy_list)
    np.save(os.path.join(write_path_iter, 'cost_test.npy'), cost_list_test)
    np.save(os.path.join(write_path_iter, 'accuracy_test.npy'), accuracy_list_test)

    for i in range(max_iter):
        if len(stop_accuracy_list) == stop_count:
            del stop_accuracy_list[0]
        ## calculate gradient
        gradient = []
        for j in range(params.shape[0]):
            if flags[int(j/3)] == 1:
                if (j + 1) % 3 == 0:
                    h = h_s
                else:
                    h = h_t
                paramPlus = params.copy()
                paramMinus = params.copy()
                paramPlus[j] += h
                paramMinus[j] -= h
                formulaPlus = update_formula(cur_formula_list, paramPlus)
                robustnessPlus = cal_robustness(time['train'], signal['train'], formulaPlus, multi_flag)
                robustnessPlus = robustnessPlus * label['train']
                costPlus = np.sum(np.exp(-1 * robustnessPlus)) + penalty_lambda * penalty_count

                formulaMinus = update_formula(cur_formula_list, paramMinus)
                robustnessMinus = cal_robustness(time['train'], signal['train'], formulaMinus, multi_flag)
                robustnessMinus = robustnessMinus * label['train']
                costMinus = np.sum(np.exp(-1 * robustnessMinus)) + penalty_lambda * penalty_count
                g = (costPlus - costMinus) / (2 * h)
            else:
                g = 0.
            gradient.append(g)
        gradient = np.array(gradient)

        ## update params
        mt = b1 * mt + (1 - b1) * gradient
        vt = b2 * vt + (1 - b2) * (gradient**2)
        mtt = mt / (1 - (b1**(i + 1)))
        vtt = vt / (1 - (b2**(i + 1)))
        vtt_sqrt = np.sqrt(vtt)
        params = params - alpha * mtt / (vtt_sqrt + e)
        cur_formula_list = update_formula(cur_formula_list, params)

        ## calculate current cost and accuracy
        robustness_res = cal_robustness(time['train'], signal['train'], cur_formula_list, multi_flag)
        robustness_res = robustness_res * label['train']
        cost = np.sum(np.exp(-1 * robustness_res)) + penalty_lambda * penalty_count
        accuracy = np.sum(robustness_res > 0.) / robustness_res.shape[0]

        robustness_res_test = cal_robustness(time['test'], signal['test'], cur_formula_list, multi_flag)
        robustness_res_test = robustness_res_test * label['test']
        cost_test = np.sum(np.exp(-1 * robustness_res_test)) + penalty_lambda * penalty_count
        accuracy_test = np.sum(robustness_res_test > 0.) / robustness_res_test.shape[0]

        cost_list.append(cost)
        accuracy_list.append(accuracy)
        cost_list_test.append(cost_test)
        accuracy_list_test.append(accuracy_test)

        write_path_iter = os.path.join(write_path, f'iter_{i}')
        os.makedirs(write_path_iter, exist_ok=True)
        np.save(os.path.join(write_path_iter, 'formula.npy'), cur_formula_list)
        np.save(os.path.join(write_path_iter, 'cost.npy'), cost_list)
        np.save(os.path.join(write_path_iter, 'accuracy.npy'), accuracy_list)
        np.save(os.path.join(write_path_iter, 'cost_test.npy'), cost_list_test)
        np.save(os.path.join(write_path_iter, 'accuracy_test.npy'), accuracy_list_test)
        draw_res(write_path_iter, cost_list, accuracy_list, cost_list_test, accuracy_list_test)

        print(f'Train\titer: {i}\tcost: {cost}\taccuracy: {accuracy}')
        print(f'Test\titer: {i}\taccuracy: {accuracy_test}')
        with open(os.path.join(write_path, f'info.txt'), 'a') as ftxt:
            ftxt.write(f'Train\titer: {i}\tcost: {cost}\taccuracy: {accuracy}\n')
            ftxt.write(f'Test\titer: {i}\taccuracy: {accuracy_test}\n')
        top_acc = np.max(accuracy_list_test)
        stop_accuracy_list.append(accuracy_test)
        if len(stop_accuracy_list) == stop_count and top_acc > 0.85 and judge(stop_accuracy_list):
            print(f'break because of accuracy did not grow for {stop_count} iterations')
            break
        if acc_threshold:
            if accuracy >= acc_threshold:
                print('break because of accuracy threshold')
                break
    top_acc_cost = cost_list[accuracy_list_test.index(top_acc)]
    top_acc_iter = accuracy_list_test.index(top_acc) - 1
    for i in range(len(accuracy_list_test)):
        if accuracy_list_test[i] == top_acc:
            if cost_list[i] < top_acc_cost:
                top_acc_cost = cost_list[i]
                top_acc_iter = i - 1
    if top_acc_iter == -1:
        top_acc_iter = 'init'
    best_formula_list = np.load(os.path.join(write_path, f'iter_{top_acc_iter}', 'formula.npy')).tolist()
    with open(os.path.join(write_path, f'info.txt'), 'a') as ftxt:
        ftxt.write('\n**********************************************\n')
        ftxt.write(f'Top Test Accuracy: {top_acc} (iteration {top_acc_iter})\n')
        ftxt.write(f'Top Formula: {best_formula_list}\n')
    return best_formula_list, top_acc, top_acc_iter
