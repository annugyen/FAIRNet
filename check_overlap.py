import json

layer_dict = {
    "Convolution1D": "Conv1D",
    "Convolution2D": "Conv2D",
    "Convolution3D": "Conv3D"
}

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
                        layer_h5_name = layers_h5[k]['layer_type']
                        layer_code_name = layers_code[str(int(k) + 1)]['name']
                        if layer_h5_name != layer_code_name:
                            if layer_dict.get(layer_code_name) != layer_h5_name:
                                break
                    else:
                        #if optimizer_h5 == optimizer_code and loss_h5 == loss_code and metrics_h5 == metrics_code:
                        has_overlap.append('h5_' + j + '=' + 'code_' + i)
    else:
        has_overlap = 'Error'
    return has_overlap

if __name__ == '__main__':
    #data_path = './data.json'
    result_code_path = './result_data.json'
    result_h5_path = './result_data_h5_merged.json'
    result_overlap_path = './result_data_overlap.json'
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
    