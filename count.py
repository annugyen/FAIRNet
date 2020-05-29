import json

from check_overlap import trans_loss

result_code_path = './result_data_v4.json'
result_h5_path = './result_data_h5_merged.json'

with open(result_code_path, 'r') as f:
    result_code_json = json.load(f)
f.close()

with open(result_h5_path, 'r') as f:
    result_h5_json = json.load(f)
f.close()

opti_list = []
loss_list = []
layer_list = []

#'''
for idx in result_code_json:
    models = result_code_json[idx]['models']
    if isinstance(models, dict):
        for model_idx in models:
            if models[model_idx]:
                opti_list.append(models[model_idx].get('compile_info', {}).get('optimizer', ''))
                loss_list.append(models[model_idx].get('compile_info', {}).get('loss', ''))
            layers = models[model_idx].get('layers', {})
            if len(layers) > 0:
                for layer_idx in layers:
                    layer_list.append(layers[layer_idx].get('name'))
                    if layers[layer_idx].get('name') == 'dense5':
                        a = 1
            

'''
for idx in result_h5_json:
    models = result_h5_json[idx]['models']
    if isinstance(models, dict):
        for model_idx in models:
            if models[model_idx]:
                opti_list.append(models[model_idx].get('compile_info', {}).get('optimizer', ''))
                loss_list.append(models[model_idx].get('compile_info', {}).get('loss', ''))
'''

loss_list_2 = []
for loss in loss_list:
    if loss not in loss_list_2:
        loss_list_2.append(loss)

loss_list_3 = [trans_loss(x) for x in loss_list_2]

trans_loss('categorical_hinge')

a = 1
