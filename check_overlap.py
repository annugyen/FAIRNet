import json
from copy import deepcopy

layer_dict = {
    "Convolution1D": "Conv1D",
    "Convolution2D": "Conv2D",
    "Convolution3D": "Conv3D",
    "MaxPool1D": "MaxPooling1D",
    "MaxPool2D": "MaxPooling2D",
    "MaxPool3D": "MaxPooling3D",
    "AvgPool1D": "AveragePooling1D",
    "AvgPool2D": "AveragePooling2D",
    "AvgPool3D": "AveragePooling3D",
    "GlobalMaxPool1D":"GlobalMaxPooling1D",
    "GlobalMaxPool2D": "GlobalMaxPooling2D",
    "GlobalMaxPool3D": "GlobalMaxPooling3D",
    "GlobalAvgPool1D": "GlobalAveragePooling1D",
    "GlobalAvgPool2D": "GlobalAveragePooling2D",
    "GlobalAvgPool3D": "GlobalAveragePooling3D",
    "InputLayer": "Input",
}

opti_list = []
loss_list = []

def trans_opti(optimizer):
    opti = str(optimizer).lower()
    if 'sgd' in opti:
        return 'SGD'
    elif 'nada' in opti:
        return 'Nadam'
    elif 'adamax' in opti:
        return 'Adamax'
    elif 'adag' in opti:
        return 'Adagrad'
    elif 'adade' in opti:
        return 'Adadelta'
    elif 'ada' in opti:
        return 'Adam'
    elif 'rms' in opti:
        return 'RMSprop'
    elif 'ftrl' in opti:
        return 'Ftrl'
    else:
        return optimizer

def trans_loss(loss):
    l = str(loss).lower()
    if 'categorical' in l:
        if 'sparse' in l:
            return 'sparsecategoricalcrossentropy'
        elif 'hinge'  in l:
            return 'categoricalhinge'
        else:
            return 'categoricalcrossentropy'
    elif 'binary' in l:
        return 'binarycrossentropy'
    elif 'mae' in l:
        return 'meanabsoluteerror'
    elif 'mse' in l:
        return 'meansquarederror'
    elif 'mean' in l:
        if 'square' in l:
            if 'logar' in l:
                return 'meansquaredlogarithmicerror'
            else:
                return 'meansquarederror'
        elif 'absolut' in l:
            if 'percent' in l:
                return 'meanabsolutepercentageerror'
            else:
                return 'meanabsoluteerror'
    elif 'mape' in l:
        return 'meanabsolutepercentageerror'
    elif 'logcosh' in l:
        return 'logcosh'
    elif 'huber' in l:
        return 'huberloss'
    elif 'hinge' in l:
        if 'square' in l:
            return 'squaredhinge'
        elif 'categorical' in l:
            return 'categoricalhinge'
        else:
            return 'hinge'
    elif 'cosine' in l:
        return 'cosineproximity'
    elif 'kullback' in l:
        return 'kullbackleiblerdivergence'
    else:
        return loss
    
    
def check_overlap(models_code, models_h5):
    has_overlap = []
    if isinstance(models_code, dict) and isinstance(models_h5, dict) and len(models_code) > 0 and len(models_h5) > 0:
        for j in models_h5:
            offset_h5 = 0
            model_h5 = models_h5[j]
            layers_h5 = deepcopy(model_h5['layers'])
            if layers_h5.get('0', {}).get('layer_type') == 'InputLayer':
                del layers_h5['0']
                offset_h5 = -1
            optimizer_h5 = model_h5['compile_info']['optimizer']
            loss_h5 = model_h5['compile_info']['loss']
            metrics_h5 =  model_h5['compile_info']['metrics']
            for i in models_code:
                offset_code = 0
                model_code = models_code[i]
                layers_code = deepcopy(model_code['layers'])
                if layers_code.get('1', {}).get('layer_type') == 'InputLayer':
                    del layers_code['1']
                    offset_code = 1
                optimizer_code = model_code.get('compile_info', {}).get('optimizer', [])
                loss_code = model_code.get('compile_info', {}).get('loss', [])
                metrics_code =  model_code.get('compile_info', {}).get('metrics', [])
                if len(layers_code) == len(layers_h5) and len(layers_code) > 0:
                    for k in layers_h5:
                        layer_h5_type = layers_h5[k]['layer_type']
                        layer_h5_name = layers_h5[k]['layer_name']
                        layer_code_name = layers_code[str(int(k) + 1 + offset_h5 + offset_code)]['layer_type']
                        if layer_h5_type != layer_code_name:
                            if layer_dict.get(layer_code_name) != layer_h5_type and layer_h5_name != layer_code_name:
                                break
                    else:
                        #opti_list.append(str(optimizer_code)+'||'+str(optimizer_h5))
                        #loss_list.append(str(loss_code)+'||'+str(loss_h5))
                        #if optimizer_h5 == optimizer_code and loss_h5 == loss_code and metrics_h5 == metrics_code:
                        if optimizer_h5 == trans_opti(optimizer_code) and trans_loss(loss_h5) == trans_loss(loss_code):
                            has_overlap.append('h5_' + j + '=' + 'code_' + i)
    else:
        has_overlap = 'Error'
    return has_overlap

if __name__ == '__main__':
    #data_path = './data.json'
    result_code_path = './result_data_v7.json'
    result_h5_path = './result_data_h5_merged.json'
    result_overlap_path = './result_data_overlap_v5.json'
    '''
    with open(data_path, 'r') as f:
        data_json = json.load(f)
    f.close()
    '''

    with open(result_code_path, 'r') as f:
        result_code_json = json.load(f)
    f.close()

    with open(result_h5_path, 'r') as f:
        result_h5_json = json.load(f)
    f.close()

    result_overlap_dict = {}
    for idx in result_h5_json:
        has_overlap = check_overlap(result_code_json[idx]['models'], result_h5_json[idx]['models'])
        result_overlap_dict[idx] = {
            'repo_full_name': result_code_json[idx]['repo_full_name'],
            'has_overlap': has_overlap
        }
    
    result_overlap_json = json.dumps(result_overlap_dict, indent = 4, separators = (',', ': '))
    with open(result_overlap_path, 'w') as f:
        f.write(result_overlap_json)
    f.close()
    
    a = 1