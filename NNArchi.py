import json
import random
import re
import time
from urllib import request

from ast_etree import extract_architecture_from_python_ast

global Token_list
Token_list = ['1498e12f29400f246eb5b2cf1c6ccfb1a4970c4a',
              'f24e286bab8a40e1995c1f1dcacd6e31e6816ac1',
              'd10d26ffe911ab0d569b5ddc5c777f26b9ef057a',
              '66b9970c6a9d24d61e167cda2f3997219bae3447']

def get_repo_full_name(repo_url):
    name_pattern = re.compile('github\.com/', re.I)
    name_search = name_pattern.search(repo_url)
    repo_full_name = repo_url[name_search.end():]
    return repo_full_name

def extract_architecture_from_python(repo_full_name):
    """
    Extract architecture of NNs from .py file in a repo.
    """

    search_terms = ['"import+keras"', '"from+keras"', 'keras.models', 'keras.layers', 'keras.utils', 'tf.keras.models.Sequential()']
    query_search_terms = '+OR+'.join(search_terms)
    search_url = 'https://api.github.com/search/code?limit=100&per_page=100&q=' + query_search_terms + '+in:file+extension:py+repo:' + repo_full_name
    Token_idx = random.randint(0,len(Token_list) - 1)
    headers_1 = {'Authorization':'token ' + Token_list[Token_idx]}
    url_request = request.Request(search_url, headers = headers_1)
    try:
        response = request.urlopen(url_request)
    except Exception as e:
        json_data = {}
    else:
        html = response.read()
        json_data = json.loads(html.decode("utf-8"))

    py_files_list = []
    if 'items' in json_data:
        for file in json_data['items']:
            raw_url = file['html_url'].replace('github.com', 'raw.githubusercontent.com').replace('blob/', '')
            py_files_list.append(raw_url)
    else:
        print('Keras may not be used.')

    model_num = 0
    models_archi = {}

    for raw_file_url in py_files_list:
        Token_idx = random.randint(0,len(Token_list) - 1)
        headers_2 = {'Authorization':'token ' + Token_list[Token_idx]}
        raw_file_request = request.Request(raw_file_url, headers = headers_2)
        raw_file = request.urlopen(raw_file_request).read().decode("utf-8")
        
        libs_set = set()
        lib_search = re.finditer('^(from|import)\s(\w+)', raw_file, re.MULTILINE)
        if lib_search:
            for lib in lib_search:
                libs_set.add(lib.group(2))
        
        if ('keras' in libs_set) or ('Keras' in libs_set):
            models_archi_temp = extract_architecture_from_python_ast(raw_file, model_num)
            models_archi = {**models_archi, **models_archi_temp}
    return models_archi

if __name__ == '__main__':
    data_path = './filtered_data.json'
    with open(data_path, 'r') as file:
        json_data = json.load(file)
    file.close()
    repo_url_dict = json_data['repo_url']
    for idx, repo in enumerate(repo_url_dict):
        repo_url = repo_url_dict[repo]
        repo_full_name = get_repo_full_name(repo_url)
        if idx == 19:
            a = 1
        models_archi = extract_architecture_from_python(repo_full_name)
        print('%d: finish' % idx)
        time.sleep(1)