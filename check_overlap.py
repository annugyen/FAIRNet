import json

layer_dict = {
    "Convolution1D": "Conv1D",
    "Convolution2D": "Conv2D",
    "Convolution3D": "Conv3D"
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
            return 'sparse_categorical_crossentropy'
        elif 'hinge'  in l:
            return 'categorical_hinge'
        else:
            return 'categorical_crossentropy'
    elif 'binary' in l:
        return 'binary_crossentropy'
    elif 'mae' in l:
        return 'mae'
    elif 'mse' in l:
        return 'mse'
    elif 'mean' in l:
        if 'square' in l:
            if 'logar' in l:
                return 'mean_squared_logarithmic_error'
            else:
                return 'mse'
        elif 'absolut' in l:
            if 'percent' in l:
                return 'mape'
            else:
                return 'mae'
    elif 'mape' in l:
        return 'mape'
    elif 'logcosh' in l:
        return 'logcosh'
    elif 'huber' in l:
        return 'huber_loss'
    elif 'hinge' in l:
        if 'square' in l:
            return 'squared_hinge'
        elif 'categorical' in l:
            return 'categorical_hinge'
        else:
            return 'hinge'
    elif 'cosine' in l:
        return 'cosine_proximity'
    elif 'kullback' in l:
        return 'kullback_leibler_divergence'
    else:
        return loss
    
    
def check_overlap(models_code, models_h5):
    has_overlap = []
    if isinstance(models_code, dict) and isinstance(models_h5, dict) and len(models_code) > 0 and len(models_h5) > 0:
        for j in models_h5:
            model_h5 = models_h5[j]
            layers_h5 = model_h5['layers']
            optimizer_h5 = model_h5['compile_info']['optimizer']
            loss_h5 = model_h5['compile_info']['loss']
            metrics_h5 =  model_h5['compile_info']['metrics']
            for i in models_code:
                model_code = models_code[i]
                layers_code = model_code['layers']
                optimizer_code = model_code.get('compile_info', {}).get('optimizer', [])
                loss_code = model_code.get('compile_info', {}).get('loss', [])
                metrics_code =  model_code.get('compile_info', {}).get('metrics', [])
                if len(layers_code) == len(layers_h5) and len(layers_code) > 0:
                    for k in layers_h5:
                        layer_h5_type = layers_h5[k]['layer_type']
                        layer_h5_name = layers_h5[k]['layer_name']
                        layer_code_name = layers_code[str(int(k) + 1)]['name']
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
    result_code_path = './result_data_v4.json'
    result_h5_path = './result_data_h5_merged.json'
    result_overlap_path = './result_data_overlap_v2.json'
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