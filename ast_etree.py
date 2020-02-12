import ast
import json
import os

from lxml import etree
from astexport.export import export_json
from json2xml import json2xml
from json2xml.utils import readfromstring

file_path = './example.py'
with open(file_path, 'r') as file:
    data = file.read()
file.close()

code_ast = ast.parse(data)
code_json = export_json(code_ast)
code_xml = json2xml.Json2xml(readfromstring(code_json)).to_xml()

'''
#print(code_xml)
with open('./code_xml.xml', 'w') as f:
    f.write(code_xml)
file.close()
'''
'''
code_tree = etree.parse('./test_xml.xml')
books = code_tree.xpath('/bookstore/book[price>35.00]')
'''

code_tree = etree.fromstring(code_xml)
#childs = code_tree.getchildren()

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
                if expr_func.find('attr').text == 'add':
                    layer_num += 1
                    layer_name = value.xpath('child::args/item/func/id')[0].text
                    layers[layer_num] = layer_name
                a = 1
            else:
                pass
            a = 1
        models[model_num] = layers
a = 1