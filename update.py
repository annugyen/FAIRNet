import difflib
import json
import time

from model_h5 import extract_architecture_from_h5

def links_select(h5_files_links):
    selected_list = []
    selected_list.append(h5_files_links[0])
    for i in range(len(h5_files_links)):
        for j in range(len(selected_list)):
            if difflib.SequenceMatcher(None, h5_files_links[i].lower(), selected_list[j].lower()).ratio() > 0.95:
                break
        else:
            selected_list.append(h5_files_links[i])
    return selected_list

if __name__ == '__main__':
    data_path = './data.json'
    result_path = './result_data_h5.json'
    with open(data_path, 'r') as f:
        data_json = json.load(f)
    f.close()
    results = {}
    for idx in range(len(data_json)):
        data = data_json[idx]
        has_h5 = data['has_h5']
        keras_used = data['keras_used']
        if has_h5 and keras_used:
            models = {}
            model_num = 0
            error_count = 0
            h5_files_links = data['h5_files_links']
            repo_full_name = data['repo_full_name']
            result_dict = {'repo_full_name': repo_full_name}
            if len(h5_files_links) > 1:
                h5_files_links = links_select(h5_files_links)
            for link in h5_files_links:
                try:
                    extracted_architecture, loss, optimizer, metrics, layers = extract_architecture_from_h5(link)
                except Exception as e:
                    error_count += 1
                else:
                    if extracted_architecture:
                        model_num += 1
                        if metrics[0] == 'loss':
                            metrics = metrics[1:]
                        models[model_num] = {'layers': layers,
                                             'compile_info':{'optimizer': optimizer,
                                                             'loss': loss,
                                                             'metrics': metrics}}
            if (not models) and error_count > 0:
                models = 'Error'
            result_dict['models'] = models
            results[idx] = result_dict
            print('%d: finish' % idx)
            time.sleep(0.5)
        if len(results) == 10:
            break
    result_json = json.dumps(results, indent = 4, separators = (',', ': '))
    with open(result_path, 'w') as file:
        file.write(result_json)
    file.close()
    a = 1
