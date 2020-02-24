import ast
import json
import os
import re

from astexport.export import export_json
from json2xml import json2xml
from json2xml.utils import readfromstring
from lxml import etree


def rebuild_tuple(root):
    rebuilt_tuple = []
    if root.find('ast_type').text == 'Num':
        return int(root.xpath('child::n/n')[0].text)
    elif root.find('ast_type').text == 'Tuple':
        items = root.xpath('child::elts/item')
        for item in items:
            rebuilt_tuple.append(rebuild_tuple(item))
    return rebuilt_tuple

def list_to_tuple(tuple_list):
    tuple_list = str(tuple_list)
    tuple_list = re.sub('\[', '(', tuple_list)
    tuple_list = re.sub(']', ')', tuple_list)
    tuple_list = eval(tuple_list)
    return tuple_list

def get_func_call_paras_kws(root, is_def_func = False, **kwarg):
    if not is_def_func:
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
                if is_def_func:
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
                if is_def_func:
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
                func_call_kw_value = list_to_tuple(rebuild_tuple(func_call_kw.find('value')))
            else:
                pass
            func_call_kws_dict[func_call_kw_key] = func_call_kw_value
    a = 1
    return func_call_paras_list, func_call_kws_dict

file_path = './francarranza_genre_classification.py'
#file_path = './nagyben_CarND-Behavioral-Cloning-P3.py'
with open(file_path, 'r') as file:
    data = file.read()
file.close()

code_ast = ast.parse(data)
code_json = export_json(code_ast)
code_xml = json2xml.Json2xml(readfromstring(code_json)).to_xml()

#print(code_xml)
with open('./code_xml.xml', 'w') as f:
    f.write(code_xml)
file.close()

code_tree = etree.fromstring(code_xml)

models = {}
model_num = 0

seqs = code_tree.xpath('//func[id = "Sequential"]')
if seqs:
    for seq in seqs:
        model_num += 1
        layers = {}
        layer_num = 0
        seq_name = ''
        func = seq.xpath('ancestor::item[ast_type = "FunctionDef"]')
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
                    func_defaults_dict[func_def_paras_list[len(func_def_paras) - len(func_defaults) + idx]] = list_to_tuple(rebuild_tuple(func_defaults[idx]))
                else:
                    pass
        #check if func is called
        func_xpath = 'descendant::func[id="' + func_name + '"]'
        func_call = code_tree.xpath(func_xpath)
        is_func_called = False
        if len(func_call) > 0:
            is_func_called = True
            func_call_root = func_call[0].xpath('..')[0]
            func_call_paras_list, func_call_kws_dict = get_func_call_paras_kws(func_call_root, is_def_func = True)
        else:
            func_call_paras_list = []
            func_call_kws_dict = {}
        if func_def_paras_list:
            func_paras_dict = dict(zip(func_def_paras_list, func_call_paras_list))
        else:
            func_paras_dict = {}
        func_paras_kws_dict = {**func_paras_dict, **func_call_kws_dict}
        func_details = func[0].find('body').getchildren()
        for func_detail in func_details:
            ast_type = func_detail.find('ast_type')
            if ast_type.text == 'Assign':
                seq_target = func_detail.find('targets')
                seq_name = seq_target.xpath('descendant::id')[0].text
                a = 1
            elif ast_type.text == 'Expr':
                value = func_detail.find('value')
                expr_func = value.find('func')
                if expr_func.find('attr').text == 'add' and expr_func.xpath('child::value/id')[0].text == seq_name:
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
                    """
                    if len(layer_paras) > 0:
                        for layer_para in layer_paras:
                            layer_para_type = layer_para.find('ast_type').text
                            if layer_para_type == 'Lambda':
                                pass #todo
                            elif layer_para_type == 'Num':
                                layer_paras_list.append(layer_para.xpath('child::n/n')[0].text)
                            elif layer_para_type == 'Name':
                                layer_para_id = layer_para.find('id').text
                                if layer_para_id in func_paras_kws_dict:
                                    layer_paras_list.append(func_paras_kws_dict[layer_para_id])
                                elif layer_para_id in func_defaults_dict:
                                    layer_paras_list.append(func_defaults_dict[layer_para_id])
                                else:
                                    layer_paras_list.append('not_found') #todo find other paras
                            else:
                                pass
                        layer['parameters'] = layer_paras_list
                    layer_kws = value.xpath('child::args/item')[0].find('keywords')
                    if len(layer_kws) > 0:
                        for layer_kw in layer_kws:
                            layer_kw_key = layer_kw.xpath('child::arg')[0].text
                            layer_kw_value_type = layer_kw.xpath('child::value/ast_type')[0].text
                            layer_kw_value = None
                            if layer_kw_value_type == 'Num':
                                layer_kw_value = layer_kw.xpath('child::value/n/n')[0].text
                            elif layer_kw_value_type == 'Name':
                                layer_kw_value = layer_kw.xpath('child::value/id')[0].text
                                if layer_kw_value in func_paras_kws_dict:
                                    layer_kw_value = func_paras_kws_dict[layer_kw_value]
                                    a = 1
                                elif layer_kw_value in func_defaults_dict:
                                    layer_kw_value = func_defaults_dict[layer_kw_value]
                                else:
                                    layer_kw_value = 'not_found' #todo find other para
                            elif layer_kw_value_type == 'Str':
                                layer_kw_value = layer_kw.xpath('child::value/s')[0].text
                            elif layer_kw_value_type == 'Tuple':
                                layer_kw_value = list_to_tuple(rebuild_tuple(layer_kw.find('value')))
                            else:
                                pass
                            layer[layer_kw_key] = layer_kw_value
                            a = 1
                    """
                    layers[layer_num] = layer
                a = 1
            else:
                pass
            a = 1
        models[model_num] = layers
a = 1
