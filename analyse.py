import json

result_path = './result_data_v6.json'
data_path = './data.json'

with open(result_path, 'r') as f:
    results_data = json.load(f)
f.close()

with open(data_path, 'r') as f:
    data_json = json.load(f)
f.close()

empty_list = []
error_list = []

lambda_list = []
activi_list = []
no_keras_list = []
valid_list = []
user_list = []
license_list = []
time_list = []

for idx in range(len(results_data)):
    if not results_data[str(idx)]['models']:
        empty_list.append(results_data[str(idx)]['repo_full_name'])
    elif results_data[str(idx)]['models'] == 'Error':
        error_list.append(results_data[str(idx)]['repo_full_name'])
    elif results_data[str(idx)]['models'] == 'Keras may not be used':
        no_keras_list.append(results_data[str(idx)]['repo_full_name'])
    else:
        for model_idx in results_data[str(idx)]['models']:
            model = results_data[str(idx)]['models'][model_idx]
            for layer_idx in model['layers']:
                layer = model['layers'][layer_idx]
                if layer['layer_type'] == 'Lambda':
                    lambda_list.append(results_data[str(idx)]['repo_full_name'])
                if layer['layer_type'] == 'Activation':
                    activi_list.append(results_data[str(idx)]['repo_full_name'])
        lambda_list = list(set(lambda_list))
        activi_list = list(set(activi_list))
    models = results_data[str(idx)]['models']
    time_list.append(data_json[idx]['repo_created_at'][0:4]+data_json[idx]['repo_created_at'][5:7])
    '''
    if data_json[idx].get('license', {}) and data_json[idx].get('license', {}).get('key', ''):
        license_list.append(data_json[idx].get('license', {}).get('key', ''))
    '''
    if isinstance(models, dict) and len(models) > 0:
        if data_json[idx].get('license', {}) and data_json[idx].get('license', {}).get('key', '') and data_json[idx].get('license', {}).get('key', '').lower() != 'other':
            license_list.append(data_json[idx].get('license', {}).get('key', ''))
        user_list.append(data_json[idx].get('repo_owner', ''))
        valid_list.append(idx)
    if idx == 99:
        a = 1

'''
for key in results_data:
    if not results_data[key]['models']:
        empty_list.append(results_data[key]['repo_full_name'])
    elif results_data[key]['models'] == 'Error':
        error_list.append(results_data[key]['repo_full_name'])
    else:
        pass
'''
print('Analysis finish')
a = 1