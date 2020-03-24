import json

result_path = 'result_filtered_data.json'

with open(result_path, 'r') as file:
    results_data = json.load(file)
file.close

empty_list = []
error_list = []
lstm_list = []
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
                if layer['name'] == 'LSTM':
                    lstm_list.append(results_data[str(idx)]['repo_full_name'])
        lstm_list = list(set(lstm_list))
print('Analysis finish')