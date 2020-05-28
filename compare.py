import json
from NNArchi import extract_architecture_from_python

#data_path = './data.json'
result_path_1 = './result_data.json'
result_path_2 = './result_data_v2.json'
result_path_3 = './result_data_v3.json'
result_path_4 = './result_data_v4.json'
output_path = './output.json'

'''
with open(data_path, 'r') as f:
    data_json = json.load(f)
f.close()
'''

with open(result_path_1, 'r') as f:
    result_json_1 = json.load(f)
f.close()

with open(result_path_2, 'r') as f:
    result_json_2 = json.load(f)
f.close()

with open(result_path_3, 'r') as f:
    result_json_3 = json.load(f)
f.close()

'''
l1 = []
l2 = []
l3 = []
l4 = []
l5 = []

for idx in result_json_1:
    if result_json_1[idx]['repo_full_name'] == result_json_2[idx]['repo_full_name']:
        if isinstance(result_json_1[idx]['models'], dict):
            if isinstance(result_json_2[idx]['models'], dict):
                if len(result_json_1[idx]['models']) == 0 and len(result_json_2[idx]['models']) != 0:
                    l2.append(idx)
                elif len(result_json_1[idx]['models']) != 0 and len(result_json_2[idx]['models']) == 0:
                    l1.append(idx)
            else:
                l3.append(idx)
        else:
            if result_json_2[idx]['models'] == {}:
                l4.append(idx)
            elif result_json_1[idx]['models'] != result_json_2[idx]['models']:
                l5.append(idx)

l_diff = l1 + l3 + l4 + l5
'''

fail_list = []
no_keras_list = []
for idx in result_json_3:
    if result_json_3[idx]['models'] == 'Keras may not be used':
        no_keras_list.append(idx)


for idx in no_keras_list:
    repo_full_name = result_json_3[idx]['repo_full_name']
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
            if try_count >= 5:
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
                    if try_count >= 5:
                        fail_list.append(idx)
                        break
            else:
                print('%s:try update%d: failed' % (idx, try_count))
                if try_count >= 5:
                    fail_list.append(idx)
                    break
    if updated == True:
        result_json_3[idx] = result_dict

result_json_4 = json.dumps(result_json_3, indent = 4, separators = (',', ': '))
with open(result_path_4, 'w') as f:
    f.write(result_json_4)
f.close()

output = json.dumps(fail_list)
with open(output_path, 'w') as f:
    f.write(output)
f.close()
