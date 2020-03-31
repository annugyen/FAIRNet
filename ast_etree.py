import ast
import json
import os
import re

from astexport.export import export_json
from json2xml import json2xml
from json2xml.utils import readfromstring
from lxml import etree

global keras_apps_list
keras_apps_list = ['Xception',
                   'VGG16',
                   'VGG19',
                   'ResNet50',
                   'ResNet101',
                   'ResNet152',
                   'ResNet50V2',
                   'ResNet101V2',
                   'ResNet152V2',
                   'InceptionV3',
                   'InceptionResNetV2',
                   'MobileNet',
                   'MobileNetV2',
                   'DenseNet121',
                   'DenseNet169',
                   'DenseNet201',
                   'NASNetMobile',
                   'NASNetLarge']

def rebuild_list(root):
    rebuilt_list = []
    if root.find('ast_type').text == 'Num':
        return int(root.xpath('child::n/n')[0].text)
    elif root.find('ast_type').text == 'Str':
        return root.find('s').text
    elif root.find('ast_type').text in ['Tuple', 'List']:
        items = root.xpath('child::elts/item')
        for item in items:
            rebuilt_list.append(rebuild_list(item))
    return rebuilt_list

def list_to_tuple(tuple_list):
    tuple_list = str(tuple_list)
    tuple_list = re.sub('\[', '(', tuple_list)
    tuple_list = re.sub(']', ')', tuple_list)
    tuple_list = eval(tuple_list)
    return tuple_list

def rebuild_attr(root):
    rebuilt_attr_list = []
    while root.find('ast_type').text == 'Attribute':
        rebuilt_attr_list.append(root.find('attr').text)
        root = root.find('value')
    if root.find('ast_type').text == 'Name':
        rebuilt_attr_list.append(root.find('id').text)
    rebuilt_attr = '.'.join(reversed(rebuilt_attr_list))
    return rebuilt_attr

def get_func_call_paras_kws(root, has_ext_paras = False, **kwarg):
    if not has_ext_paras:
        func_paras_kws_dict = kwarg['func_paras_kws_dict']
        func_defaults_dict = kwarg['func_defaults_dict']
    func_call_paras = root.xpath('child::args/item')
    func_call_paras_list = []
    if len(func_call_paras) > 0:
        for func_call_para in func_call_paras:
            func_call_para_type = func_call_para.find('ast_type').text
            if func_call_para_type == 'lambda':
                pass #todo lambda
            elif func_call_para_type == 'Num':
                func_call_paras_list.append(func_call_para.xpath('child::n/n')[0].text)
            elif func_call_para_type == 'Name':
                func_call_para_name = func_call_para.find('id').text
                if has_ext_paras:
                    func_call_paras_list.append(func_call_para_name)
                elif func_call_para_name in func_paras_kws_dict:
                    func_call_paras_list.append(func_paras_kws_dict[func_call_para_name])
                elif func_call_para_name in func_defaults_dict:
                    func_call_paras_list.append(func_defaults_dict[func_call_para_name])
                else:
                    func_call_paras_list.append('not_found') #todo find other paras
            else:
                pass
    func_call_kws_dict = {}
    func_call_kws = root.find('keywords')
    if len(func_call_kws) > 0:
        for func_call_kw in func_call_kws:
            func_call_kw_key = func_call_kw.xpath('child::arg')[0].text
            func_call_kw_value_type = func_call_kw.xpath('child::value/ast_type')[0].text
            func_call_kw_value = None
            if func_call_kw_value_type == 'Num':
                func_call_kw_value = func_call_kw.xpath('child::value/n/n')[0].text
            elif func_call_kw_value_type == 'Name':
                if has_ext_paras:
                    func_call_kw_value = func_call_kw.xpath('child::value/id')[0].text
                elif func_call_kw_key in func_paras_kws_dict:
                    func_call_kw_value = func_paras_kws_dict[func_call_kw_key]
                elif func_call_kw_key in func_defaults_dict:
                    func_call_kw_value = func_defaults_dict[func_call_kw_key]
                else:
                    func_call_kw_value = 'not_found' #todo find other paras
            elif func_call_kw_value_type == 'Str':
                func_call_kw_value = func_call_kw.xpath('child::value/s')[0].text
            elif func_call_kw_value_type == 'Tuple':
                func_call_kw_value = list_to_tuple(rebuild_list(func_call_kw.find('value')))
            elif func_call_kw_value_type == 'Attribute':
                func_call_kw_value = rebuild_attr(func_call_kw.find('value'))
            elif func_call_kw_value_type == 'List':
                func_call_kw_value = rebuild_list(func_call_kw.find('value'))
            elif func_call_kw_value_type == 'Call':
                if len(func_call_kw.xpath('child::value/func/id')) > 0:
                    func_call_kw_value = func_call_kw.xpath('child::value/func/id')[0].text
                elif func_call_kw.xpath('child::value/func/ast_type')[0].text == 'Attribute':
                    func_call_kw_value = rebuild_attr(func_call_kw.xpath('child::value/func')[0])
            elif func_call_kw_value_type == 'NameConstant':
                func_call_kw_value = func_call_kw.xpath('child::value/value')[0].text
            else:
                pass
            func_call_kws_dict[func_call_kw_key] = func_call_kw_value
    return func_call_paras_list, func_call_kws_dict

def extract_architecture_from_python_ast(code_str, model_num):

    with open('./code_str.py', 'w') as f:
        f.write(code_str)
    f.close()
    
    code_ast = ast.parse(code_str)
    code_json = export_json(code_ast)
    code_xml = json2xml.Json2xml(readfromstring(code_json)).to_xml()

    #print(code_xml)
    with open('./code_xml.xml', 'w') as f:
        f.write(code_xml)
    f.close()

    code_tree = etree.fromstring(code_xml)

    models = {}
    #model_num = 0

    seqs = code_tree.xpath('//func[id = "Sequential"]')
    if len(seqs) > 0:
        for seq in seqs:
            model_num += 1
            model_name = ''
            models[model_num] = {}
            layers = {}
            compile_kws_dict = {}
            layer_num = 0
            seq_name = (seq.xpath('../..')[0]).xpath('child::targets/item/id')[0].text
            is_model_compiled = False
            func = seq.xpath('ancestor::item[ast_type = "FunctionDef"]')
            if len(func) > 0:
                model_in_func = True
                func_name = func[0].find('name').text
                func_def_paras = func[0].xpath('child::args/args')[0].getchildren()
                func_def_paras_list = []
                if len(func_def_paras) > 0:
                    for func_def_para in func_def_paras:
                        func_def_paras_list.append(func_def_para.find('arg').text)
                func_defaults = func[0].xpath('child::args/defaults')[0].getchildren()
                func_defaults_dict = {}
                if len(func_defaults) > 0:
                    for idx in range(len(func_defaults)):
                        if func_defaults[idx].find('ast_type').text == 'Num':
                            func_defaults_dict[func_def_paras_list[len(func_def_paras) - len(func_defaults) + idx]] = func_defaults[idx].xpath('child::n/n')[0].text
                        elif func_defaults[idx].find('ast_type').text == 'Tuple':
                            func_defaults_dict[func_def_paras_list[len(func_def_paras) - len(func_defaults) + idx]] = list_to_tuple(rebuild_list(func_defaults[idx]))
                        else:
                            pass
                #check if func is called
                func_xpath = 'descendant::func[id="' + func_name + '"]'
                func_call = code_tree.xpath(func_xpath)
                is_func_called = False
                if len(func_call) > 0: #todo verify each call
                    is_func_called = True
                    func_call_root = func_call[0].xpath('..')[0]
                    func_call_paras_list, func_call_kws_dict = get_func_call_paras_kws(func_call_root, has_ext_paras = True)
                    try:
                        model_name = ((func_call_root.xpath('..')[0]).xpath('child::targets/item')[0]).xpath('descendant::id')[0].text
                    except Exception as e:
                        model_name = 'not_found'
                else:
                    func_call_paras_list = []
                    func_call_kws_dict = {}
                if func_def_paras_list:
                    func_paras_dict = dict(zip(func_def_paras_list, func_call_paras_list))
                else:
                    func_paras_dict = {}
                func_paras_kws_dict = {**func_paras_dict, **func_call_kws_dict}
                func_details = func[0].find('body').getchildren()
            else:
                model_in_func = False
                func_details = code_tree.find('body')
                model_name = seq_name
                func_paras_kws_dict = {}
                func_defaults_dict = {}
            for func_detail in func_details:
                ast_type = func_detail.find('ast_type')
                if ast_type.text == 'Assign':
                    if func_detail.xpath('child::value/ast_type')[0].text == 'Call':
                        if func_detail.xpath('child::value/func/ast_type')[0].text == 'Attribute':
                            base_model_name = func_detail.xpath('child::value/func/attr')[0].text
                            if base_model_name in keras_apps_list:
                                base_model_root = func_detail.find('value')
                                base_model_paras, base_model_kws = get_func_call_paras_kws(base_model_root, func_paras_kws_dict = func_paras_kws_dict, func_defaults_dict = func_defaults_dict)
                                models[model_num]['base_model'] = {'name':base_model_name, 'parameters': base_model_paras, **base_model_kws}
                elif ast_type.text == 'Expr' and len(func_detail.xpath('child::value/func/attr')) > 0:
                    value = func_detail.find('value')
                    expr_func = value.find('func')
                    if expr_func.find('attr').text == 'add' and expr_func.xpath('child::value/id')[0].text == seq_name:
                        if len(value.xpath('child::args/item/func/id')) > 0:
                            layer_num += 1
                            layer_name = value.xpath('child::args/item/func/id')[0].text
                            layer = {}
                            layer['name'] = layer_name
                            layer_root = value.xpath('child::args/item')[0]
                            #layer_paras = value.xpath('child::args/item/args/item')
                            layer_paras_list = []
                            layer_kws_dict = {}
                            layer_paras_list, layer_kws_dict = get_func_call_paras_kws(layer_root, func_paras_kws_dict = func_paras_kws_dict, func_defaults_dict = func_defaults_dict)
                            layer['parameters'] = layer_paras_list
                            layer = {**layer, **layer_kws_dict}
                            layers[layer_num] = layer
                    elif expr_func.find('attr').text == 'compile' and expr_func.xpath('child::value/id')[0].text == seq_name:
                        is_model_compiled = True
                        compile_root = expr_func.xpath('..')[0]
                        _, compile_kws_dict = get_func_call_paras_kws(compile_root, has_ext_paras = True)
                        models[model_num]['compile_info'] = compile_kws_dict
                else:
                    pass
            models[model_num]['layers'] = layers
            if not is_model_compiled:
                model_compiles = code_tree.xpath('//func[attr = "compile"]')
                if len(model_compiles) > 0:
                    for model_compile in model_compiles:
                        if model_compile.xpath('child::value/id')[0].text == model_name:
                            is_model_compiled == True
                            compile_root = model_compile.xpath('..')[0]
                            _, compile_kws_dict = get_func_call_paras_kws(compile_root, has_ext_paras = True)
                            models[model_num]['compile_info'] = compile_kws_dict
                            break
                else:
                    models[model_num]['compile_info'] = {}
    return models, model_num

if __name__ == '__main__':
    #file_path = './francarranza_genre_classification.py'
    #file_path = './mcculzac_Volkswagen.py'
    #file_path = './nagyben_CarND-Behavioral-Cloning-P3.py'
    #file_path = './XintongHao_Self-Driving-Car-Behavioral-Cloning.py'
    file_path = './jw15_wildflower-finder.py'
    with open(file_path, 'r') as file:
        code_str = file.read()
    file.close()
    model_num = 0
    models_archi = extract_architecture_from_python_ast(code_str, model_num)
    pass