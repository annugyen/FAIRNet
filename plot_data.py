import json

import matplotlib.pyplot as plt

from check_overlap import layer_dict

C_layer_type = [
    'Conv1D',
    'Conv2D',
    'Conv2DTranspose',
    'Conv3D',
    'Conv3DTranspose',
    'Cropping1D',
    'Cropping2D',
    'Cropping3D',
    'DepthwiseConv2D',
    'SeparableConv1D',
    'SeparableConv2D',
    'UpSampling1D',
    'UpSampling2D',
    'UpSampling3D',
    'ZeroPadding1D',
    'ZeroPadding2D',
    'ZeroPadding3D',
]

C_layer_type = [l.lower() for l in C_layer_type]

R_layer_type = [
    'ConvLSTM2D',
    'ConvLSTM2DCell',
    'CuDNNGRU',
    'CuDNNLSTM',
    'GRU',
    'GRUCell',
    'LSTM',
    'LSTMCell',
    'RNN',
    'SimpleRNN',
    'SimpleRNNCell',
]

R_layer_type = [l.lower() for l in R_layer_type]

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

def get_model_type(layers, hasbasemodel):
    model_type = set()
    if hasbasemodel:
        model_type.add('cnn')
    for layer_idx in layers:
        if layers[layer_idx].get('layer_type', '').lower() in C_layer_type:
            model_type.add('cnn')
        if layers[layer_idx].get('layer_type', '').lower() in R_layer_type:
            model_type.add('rnn')
    else:
        if 'cnn' not in model_type and 'rnn' not in model_type:
            if len(layers) > 0:
                model_type.add('fnn')
            else:
                model_type.add('nn')
    return model_type

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
                    ha='center', va='bottom', fontsize='x-large')

if __name__ == '__main__':
    data_path = './data.json'
    with open(data_path, 'r') as f:
        data_json = json.load(f)
    f.close()

    result_path = './result_data_v7.json'
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

    repo_CNN_creat_time_list = []
    repo_RNN_creat_time_list = []
    repo_FNN_creat_time_list = []

    for (data, idx) in zip(data_json, result_json):
        creat_time = data['repo_created_at'][0:7]
        
        '''
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
        '''
        
        if isinstance(result_json[idx].get('models'), dict) and len(result_json[idx]['models']) > 0:
            
            '''
            if 'recurrent_type' in nn_type:
                RNN_creat_time_list.append(time_dict[creat_time])
            elif 'conv_type' in nn_type:
                CNN_creat_time_list.append(time_dict[creat_time])
            elif 'feed_forward_type' in nn_type:
                FNN_creat_time_list.append(time_dict[creat_time])
            else:
                pass
            '''

            models = result_json[idx].get('models')
            repo_type = set()
            for model_idx in models:
                layers = models[model_idx].get('layers', {})
                model_type = models[model_idx]['model_type']
                if isinstance(layers, dict) and len(layers) > 0:
                    if 'rnn' in model_type:
                        RNN_creat_time_list.append(time_dict[creat_time])
                        repo_type.add('rnn')
                    if 'cnn' in model_type:
                        CNN_creat_time_list.append(time_dict[creat_time])
                        repo_type.add('cnn')
                    if 'fnn' in model_type:
                        FNN_creat_time_list.append(time_dict[creat_time])
                        repo_type.add('fnn')
                for layer_idx in layers:
                    layer = layers[layer_idx]
                    layer_type = layer.get('layer_type', '')
                    if layer_type == 'Activation':
                        if len(layer['parameters']) > 0:
                            activation_list.append(str(layer['parameters'][0]).lower())
                    elif 'activation' in layer:
                        activation_list.append(str(layer.get('activation')).lower())

                    #CNN_layer_list.append(layers[layer_idx].get('name'))
                    if 'rnn' in model_type:
                        RNN_layer_list.append(layer_type)
                    if 'cnn' in model_type:
                        CNN_layer_list.append(layer_type)
                    if 'fnn' in model_type:
                        FNN_layer_list.append(layer_type)

            if 'rnn' in repo_type:
                repo_RNN_creat_time_list.append(time_dict[creat_time])
            if 'cnn' in repo_type:
                repo_CNN_creat_time_list.append(time_dict[creat_time])
            if 'fnn' in repo_type:
                repo_FNN_creat_time_list.append(time_dict[creat_time])

    acti_num_dict = {}
    for acti in activation_list:
        act = trans_acti(acti)
        if act != 'not_found' and act != 'none' and act != 'self.activation':
            if act not in acti_num_dict:
                acti_num_dict[act] = 1
            else:
                acti_num_dict[act] += 1
        if act == 'none':
            if 'linear' not in acti_num_dict:
                acti_num_dict['linear'] = 1
            else:
                acti_num_dict['linear'] += 1

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
    plt.xticks(range(0, 54, 4), time_axis[::4], rotation=70, fontsize='x-large')
    plt.yticks(fontsize='x-large')
    #fig1.xaxis.set_major_locator(ticker.MultipleLocator(3))
    plt.subplots_adjust(left=0.10, bottom=0.20, right=0.99, top=0.99)
    ax1.legend(loc = 'upper left', fontsize='x-large')
    #ax1.set_title('Cumulative histogram of the number of Neural Networks \n created over time in the Dataset.')
    ax1.set_xlabel('Date', fontsize='xx-large')
    ax1.set_ylabel('Number of occurrence', fontsize='xx-large')
    #fig1.savefig('./figures/cumulativeNeuralNetwork050620_1.pdf', dpi=300, format='pdf')

    fig6 = plt.figure('Figure6',figsize = (10, 6))
    plt.figure('Figure6')
    ax6 = plt.subplot(111)
    plt.hist(repo_CNN_creat_time_list, bins = 54, color = 'blue', histtype = 'step', cumulative = True, label = 'CNN')
    plt.hist(repo_RNN_creat_time_list, bins = 54, color = 'red', histtype = 'step', cumulative = True, label = 'RNN')
    plt.hist(repo_FNN_creat_time_list, bins = 54, color = 'green', histtype = 'step', cumulative = True, label = 'FNN')
    plt.xticks(range(0, 54, 4), time_axis[::4], rotation=70, fontsize='x-large')
    plt.yticks(fontsize='x-large')
    plt.subplots_adjust(left=0.10, bottom=0.20, right=0.99, top=0.99)
    ax6.legend(loc = 'upper left', fontsize='x-large')
    #ax6.set_title('Cumulative histogram of the number of Neural Networks \n created over time in the Dataset.')
    ax6.set_xlabel('Date', fontsize='xx-large')
    ax6.set_ylabel('Number of occurrence', fontsize='xx-large')
    #fig6.savefig('./figures/cumulativeNeuralNetwork050620_2.pdf', dpi=300, format='pdf')

    num_repo_CNN_creat_time_list = [repo_CNN_creat_time_list.count(t) for t in range(54)]
    sum_repo_CNN_creat_time_list = [sum(num_repo_CNN_creat_time_list[:t]) for t in range(1, 55)]
    num_repo_RNN_creat_time_list = [repo_RNN_creat_time_list.count(t) for t in range(54)]
    sum_repo_RNN_creat_time_list = [sum(num_repo_RNN_creat_time_list[:t]) for t in range(1, 55)]
    num_repo_FNN_creat_time_list = [repo_FNN_creat_time_list.count(t) for t in range(54)]
    sum_repo_FNN_creat_time_list = [sum(num_repo_FNN_creat_time_list[:t]) for t in range(1, 55)]

    fig7 = plt.figure('Figure7',figsize = (10, 6))
    plt.figure('Figure7')
    ax7 = plt.subplot(111)
    plt.plot(range(54), sum_repo_CNN_creat_time_list, 'bv-', label = 'CNN')
    plt.plot(range(54), sum_repo_RNN_creat_time_list, 'rs-', label = 'RNN')
    plt.plot(range(54), sum_repo_FNN_creat_time_list, 'go-', label = 'FNN')
    plt.xticks(range(0, 54, 4), time_axis[::4], rotation=70, fontsize='x-large')
    plt.yticks(fontsize='x-large')
    plt.subplots_adjust(left=0.10, bottom=0.20, right=0.99, top=0.99)
    ax7.legend(loc = 'upper left', fontsize='x-large')
    #ax7.set_title('Cumulative histogram of the number of Neural Networks \n created over time in the Dataset.')
    ax7.set_xlabel('Date', fontsize='xx-large')
    ax7.set_ylabel('Number of occurrence', fontsize='xx-large')
    fig7.savefig('./figures/cumulativeNeuralNetwork050620_3.pdf', dpi=300, format='pdf')
    
    fig8 = plt.figure('Figure8',figsize = (10, 6))
    plt.figure('Figure8')
    ax8 = plt.subplot(111)
    plt.plot(range(54), sum_repo_CNN_creat_time_list, 'b--', drawstyle='steps-mid', label = 'CNN')
    #plt.hist(repo_CNN_creat_time_list, bins = 54, color = 'blue', histtype = 'step', cumulative = True, label = 'CNN1')
    plt.plot(range(54), sum_repo_RNN_creat_time_list, 'r-.', drawstyle='steps-mid', label = 'RNN')
    plt.plot(range(54), sum_repo_FNN_creat_time_list, 'g-', drawstyle='steps-mid', label = 'FNN')
    plt.xticks(range(0, 54, 4), time_axis[::4], rotation=70, fontsize='x-large')
    plt.yticks(fontsize='x-large')
    plt.subplots_adjust(left=0.10, bottom=0.20, right=0.99, top=0.99)
    ax8.legend(loc = 'upper left', fontsize='x-large')
    #ax8.set_title('Cumulative histogram of the number of Neural Networks \n created over time in the Dataset.')
    ax8.set_xlabel('Date', fontsize='xx-large')
    ax8.set_ylabel('Number of occurrence', fontsize='xx-large')
    #fig8.savefig('./figures/cumulativeNeuralNetwork050620_4.pdf', dpi=300, format='pdf')    
    
    plt.show()

    #draw bar plot of activation functions
    fig2 = plt.figure('Figure2',figsize = (10, 6))
    plt.figure('Figure2')
    ax2 = plt.subplot(111)
    rects2 = plt.bar(acti_name_list[:10], acti_num_list[:10])
    autolabel(rects2, ax2)
    plt.xticks(rotation = 70, fontsize='xx-large')
    plt.yticks(fontsize='x-large')
    plt.subplots_adjust(left=0.11, bottom=0.20, right=0.99, top=0.995)
    #ax2.set_title('Top 10 used activation functions in the neural networks')
    ax2.set_ylabel('Number of Activation Functions', fontsize='xx-large')
    #fig2.savefig('./figures/activationFunction050620.pdf', dpi=300, format='pdf')

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

    #plt.show()
