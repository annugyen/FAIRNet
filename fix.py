import json
import time
from copy import deepcopy

from NNArchi import extract_architecture_from_python
from plot_data import C_layer_type, R_layer_type, get_model_type

trans_layer_dict = {
    "Convolution1D": "Conv1D",
    "Convolution2D": "Conv2D",
    "Convolution3D": "Conv3D",
    "InputLayer": "Input",
}

all_layer_type_list = [
    'Activation',
    'ActivityRegularization',
    'AveragePooling1D',
    'AveragePooling2D',
    'AveragePooling3D',
    'BatchNormalization',
    'Conv1D',
    'Conv2D',
    'Conv2DTranspose',
    'Conv3D',
    'Conv3DTranspose',
    'ConvLSTM2D',
    'ConvLSTM2DCell',
    'Cropping1D',
    'Cropping2D',
    'Cropping3D',
    'CuDNNGRU',
    'CuDNNLSTM',
    'Dense',
    'DepthwiseConv2D',
    'Dropout',
    'Embedding',
    'Flatten',
    'GRU',
    'GRUCell',
    'GlobalAveragePooling1D',
    'GlobalAveragePooling2D',
    'GlobalAveragePooling3D',
    'GlobalMaxPooling1D',
    'GlobalMaxPooling2D',
    'GlobalMaxPooling3D',
    'Input',
    'LSTM',
    'LSTMCell',
    'Lambda',
    'LocallyConnected1D',
    'LocallyConnected2D',
    'Masking',
    'MaxPooling1D',
    'MaxPooling2D',
    'MaxPooling3D',
    'Permute',
    'RNN',
    'RepeatVector',
    'Reshape',
    'SeparableConv1D',
    'SeparableConv2D',
    'SimpleRNN',
    'SimpleRNNCell',
    'SpatialDropout1D',
    'SpatialDropout2D',
    'SpatialDropout3D',
    'UpSampling1D',
    'UpSampling2D',
    'UpSampling3D',
    'ZeroPadding1D',
    'ZeroPadding2D',
    'ZeroPadding3D',
]

all_layer_type_list = [l.lower() for l in all_layer_type_list]

'''
def fix_result():
    fix_set = set()

    for idx in result_json_5:
        models = result_json_5[idx].get('models', {})
        if isinstance(models, dict) and len(models) > 0:
            for model_idx in models:
                model = models[model_idx]
                layers = model.get('layers', {})
                for layer_idx in layers:
                    layer = layers[layer_idx]
                    layer['layer_type'] = str(layer['name'])
                    if trans_layer_dict.get(str(layer['name']), str(layer['name'])).lower() not in all_layer_type_list:
                        fix_set.add(idx)                
                    del layer['name']

    fix_list = sorted(list(fix_set))
    fail_list= []

    for idx in fix_list:
        repo_full_name = result_json_5[idx]['repo_full_name']
        result_dict = {'repo_full_name': repo_full_name}
        updated = False
        try_count = 0
        while True:
            try_count += 1
            try:
                models_archi = extract_architecture_from_python(repo_full_name)
            except Exception as e:
                result_dict['models'] = 'Error'
                print('%s:try update%d: failed' % (idx, try_count))
                if try_count >= 4:
                    fail_list.append(idx)
                    break
            else:
                print('%s:try update%d' % (idx, try_count))
                if isinstance(models_archi, dict):
                    if len(models_archi) > 0:
                        print('%s:try update%d: succeeded' % (idx, try_count))
                        result_dict['models'] = models_archi
                        updated = True
                        break
                    else:
                        print('%s:try update%d: failed' % (idx, try_count))
                        if try_count >= 4:
                            fail_list.append(idx)
                            break
                else:
                    print('%s:try update%d: failed' % (idx, try_count))
                    if try_count >= 4:
                        fail_list.append(idx)
                        break
            time.sleep(0.5)
        if updated == True:
            result_json_5[idx] = result_dict

    print(fail_list)
'''

def add_model_type():
    for idx in result_json_7:
        models = result_json_7[idx].get('models', {})
        if isinstance(models, dict) and len(models) > 0:
            for model_idx in models:
                model = models[model_idx]
                layers = model.get('layers', {})
                hasbasemodel = True if model.get('base_model') else False
                model_type = get_model_type(layers, hasbasemodel)
                models[model_idx]['model_type'] = list(model_type)

if __name__ == '__main__':
    #result_path_4 = './result_data_v4.json'
    #result_path_5 = './result_data_v5.json'
    result_path_6 = './result_data_v6.json'
    result_path_7 = './result_data_v7.json'

    with open(result_path_6, 'r') as f:
        result_json_6 = json.load(f)
    f.close()

    result_json_7 = deepcopy(result_json_6)
    add_model_type()
    result_json = json.dumps(result_json_7, indent = 4, separators = (',', ': '))
    with open(result_path_7, 'w') as f:
        f.write(result_json)
    f.close()

a = 1
