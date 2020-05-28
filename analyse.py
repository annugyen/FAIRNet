import json

result_path = './result_data_v4.json'

with open(result_path, 'r') as file:
    results_data = json.load(file)
file.close

empty_list = []
error_list = []

lambda_list = []
activi_list = []
no_keras_list = []
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
                if layer['name'] == 'Lambda':
                    lambda_list.append(results_data[str(idx)]['repo_full_name'])
                if layer['name'] == 'Activation':
                    activi_list.append(results_data[str(idx)]['repo_full_name'])
        lambda_list = list(set(lambda_list))
        activi_list = list(set(activi_list))

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