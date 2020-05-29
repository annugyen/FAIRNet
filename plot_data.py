import json

import matplotlib.pyplot as plt

from check_overlap import layer_dict

def trans_acti(acti):
    if 'softmax' in acti:
        return 'softmax'
    elif 'leakyrelu' in acti:
        return 'leakyrelu'
    elif 'relu' in acti:
        return 'relu'
    elif 'selu' in acti:
        return 'selu'
    elif 'elu' in acti:
        return 'elu'
    elif 'sigmoid' in acti:
        return 'sigmoid'
    elif 'exponential' in acti:
        return 'exponential'
    elif 'exp' in acti:
        return 'exponential'
    elif 'linear' in acti:
        return 'linear'
    elif 'softplus' in acti:
        return 'softplus'
    elif 'softsign' in acti:
        return 'softsign'
    else:
        return acti

def DictSort(Dict):
    NameList, NumList = [], []
    sortedList = sorted(Dict.items(), key = lambda item : item[1], reverse = True)
    for element in sortedList:
        NameList.append(element[0])
        NumList.append(element[1])
    return NameList, NumList

#autolabel method from official matplotlid example
def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 1),  # 1 point vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

data_path = './data.json'
with open(data_path, 'r') as f:
    data_json = json.load(f)
f.close()

result_path = './result_data_v4.json'
with open(result_path, 'r') as f:
    result_json = json.load(f)
f.close()

#generate a int list which represent months from 2015-01 to 2019-06
time_axis = []
for t in range(54):
    if (t % 12) + 1 < 10:
        t_str = str((t // 12) + 2015) + '-0' + str((t % 12) + 1)
    else:
        t_str = str((t // 12) + 2015) + '-' + str((t % 12) + 1)
    time_axis.append(t_str)

#CNN_creat_time_num = {t:0 for t in time_axis}
#RNN_creat_time_num = {t:0 for t in time_axis}
#FNN_creat_time_num = {t:0 for t in time_axis}
time_dict = {}
for i, t in enumerate(time_axis):
    time_dict[t] = i

CNN_creat_time_list = []
RNN_creat_time_list = []
FNN_creat_time_list = []
activation_list = []
CNN_layer_list = []
RNN_layer_list = []
FNN_layer_list = []

for (data, idx) in zip(data_json, result_json):
    creat_time = data['repo_created_at'][0:7]

    nn_type = None
    if 'nn_type' in data:
        nn_type = data['nn_type']
    elif 'suggested_type' in data:
        nn_type = data['suggested_type']
    else:
        nn_type = ['nn']
    
    if 'recurrent_type' in nn_type:
        RNN_creat_time_list.append(time_dict[creat_time])
    elif 'conv_type' in nn_type:
        CNN_creat_time_list.append(time_dict[creat_time])
    elif 'feed_forward_type' in nn_type:
        FNN_creat_time_list.append(time_dict[creat_time])
    else:
        pass
    
    if isinstance(result_json[idx].get('models'), dict):
        models = result_json[idx].get('models')
        for model_idx in models:
            layers = models[model_idx].get('layers', {})
            for layer_idx in layers:
                layer = layers[layer_idx]
                if layer['name'] == 'Activation':
                    if len(layer['parameters']) > 0:
                        activation_list.append(str(layer['parameters'][0]).lower())
                elif 'activation' in layer:
                    activation_list.append(str(layer.get('activation')).lower())
                CNN_layer_list.append(layers[layer_idx].get('name'))
                if 'recurrent_type' in nn_type:
                    RNN_layer_list.append(layers[layer_idx].get('name'))
                elif 'conv_type' in nn_type:
                    CNN_layer_list.append(layers[layer_idx].get('name'))
                elif 'feed_forward_type' in nn_type:
                    FNN_layer_list.append(layers[layer_idx].get('name'))
                else:
                    pass

acti_num_dict = {}
for acti in activation_list:
    act = trans_acti(acti)
    if act not in acti_num_dict:
        acti_num_dict[act] = 0
    else:
        acti_num_dict[act] += 1

acti_name_list, acti_num_list = DictSort(acti_num_dict)

CNN_layer_num_dict = {}
for layer in CNN_layer_list:
    l = layer_dict.get(layer, layer)
    if l not in CNN_layer_num_dict:
        CNN_layer_num_dict[l] = 0
    else:
        CNN_layer_num_dict[l] += 1

CNN_layer_name_list, CNN_layer_num_list = DictSort(CNN_layer_num_dict)

RNN_layer_num_dict = {}
for layer in RNN_layer_list:
    l = layer_dict.get(layer, layer)
    if l not in RNN_layer_num_dict:
        RNN_layer_num_dict[l] = 0
    else:
        RNN_layer_num_dict[l] += 1

RNN_layer_name_list, RNN_layer_num_list = DictSort(RNN_layer_num_dict)

FNN_layer_num_dict = {}
for layer in FNN_layer_list:
    l = layer_dict.get(layer, layer)
    if l not in FNN_layer_num_dict:
        FNN_layer_num_dict[l] = 0
    else:
        FNN_layer_num_dict[l] += 1

FNN_layer_name_list, FNN_layer_num_list = DictSort(FNN_layer_num_dict)

#CNN_creat_time_list = [CNN_creat_time_num.get(t) for t in CNN_creat_time_num]
#RNN_creat_time_list = [RNN_creat_time_num.get(t) for t in RNN_creat_time_num]
#FNN_creat_time_list = [FNN_creat_time_num.get(t) for t in FNN_creat_time_num]

#draw cumulative histogram of NNs
fig1 = plt.figure('Figure1',figsize = (10, 6))
plt.figure('Figure1')
ax1 = plt.subplot(111)
plt.hist(CNN_creat_time_list, bins = 54, color = 'blue', histtype = 'step', cumulative = True, label = 'CNN')
plt.hist(RNN_creat_time_list, bins = 54, color = 'red', histtype = 'step', cumulative = True, label = 'RNN')
plt.hist(FNN_creat_time_list, bins = 54, color = 'green', histtype = 'step', cumulative = True, label = 'FNN')
plt.xticks(range(54), time_axis, rotation=90)
#fig1.xaxis.set_major_locator(ticker.MultipleLocator(3))
plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.2)
ax1.legend(loc = 'upper left')
ax1.set_title('Cumulative histogram of the number of Neural Networks \n created over time in the Dataset.')
ax1.set_xlabel('Date')
ax1.set_ylabel('Number of occurrence')

#draw bar plot of activation functions
fig2 = plt.figure('Figure2',figsize = (10, 6))
plt.figure('Figure2')
ax2 = plt.subplot(111)
rects2 = plt.bar(acti_name_list[:10], acti_num_list[:10])
autolabel(rects2, ax2)
plt.xticks(rotation = 70)
plt.subplots_adjust(left = 0.15, right = 0.9, top = 0.9, bottom = 0.3)
ax2.set_title('Top 10 used activation functions in the neural networks')
ax2.set_ylabel('Number of Activation Functions')

#draw bar plot of layer types used in CNNs
fig3 = plt.figure('Figure3',figsize = (10, 6))
plt.figure('Figure3')
ax3 = plt.subplot(111)
rects3 = plt.bar(CNN_layer_name_list[:10], CNN_layer_num_list[:10])
autolabel(rects3, ax3)
plt.xticks(rotation = 70)
plt.subplots_adjust(left = 0.15, right = 0.9, top = 0.9, bottom = 0.3)
ax3.set_title('Number of Top10 used layer types in CNNs')
ax3.set_ylabel('Number of Layer Types')

#draw bar plot of layer types used in RNNs
fig4 = plt.figure('Figure4',figsize = (10, 6))
plt.figure('Figure4')
ax4 = plt.subplot(111)
rects4 = plt.bar(RNN_layer_name_list[:10], RNN_layer_num_list[:10])
autolabel(rects4, ax4)
plt.xticks(rotation = 70)
plt.subplots_adjust(left = 0.15, right = 0.9, top = 0.9, bottom = 0.3)
ax4.set_title('Number of Top10 used layer types in RNNs')
ax4.set_ylabel('Number of Layer Types')

#draw bar plot of layer types used in FNNs
fig5 = plt.figure('Figure5',figsize = (10, 6))
plt.figure('Figure5')
ax5 = plt.subplot(111)
rects5 = plt.bar(FNN_layer_name_list[:10], FNN_layer_num_list[:10])
autolabel(rects5, ax5)
plt.xticks(rotation = 70)
plt.subplots_adjust(left = 0.15, right = 0.9, top = 0.9, bottom = 0.3)
ax5.set_title('Number of Top10 used layer types in FNNs')
ax5.set_ylabel('Number of Layer Types')

plt.show()
