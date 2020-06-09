# FAIRnets

<font size=3>A framework for making neural networks FAIR</font>

This framework extracts the neural network architecture information to build a knowledge graph which is implemented in the [FAIRnets Search](https://km.aifb.kit.edu/services/fairnets/).  

## Authors

* **[Anna Nguyen](https://www.aifb.kit.edu/web/Anna_Nguyen)**
* **[Tobias Weller](https://www.aifb.kit.edu/web/Tobias_Weller)**
* **[Michael Färber](https://www.aifb.kit.edu/web/Michael_Färber)**
* **[York Sure-Vetter](https://www.aifb.kit.edu/web/York_Sure-Vetter)**
* **[Yanglin Xu](https://www.aifb.kit.edu/web/Xu,_Yanglin)**

## Paper

[Nguyen, Anna; Weller, Tobias; Färber, Michael; Sure-Vetter, York. "Making Neural Networks FAIR." *corr/abs-1907-11569.* ](https://arxiv.org/abs/1907.11569)
```
@article{DBLP:journals/corr/abs-1907-11569,
  author    = {Anna Nguyen and
               Tobias Weller and
               York Sure{-}Vetter},
  title     = {Making Neural Networks {FAIR}},
  journal   = {CoRR},
  volume    = {abs/1907.11569},
  year      = {2019},
  url       = {http://arxiv.org/abs/1907.11569}
}
```

# NNArchi Documentation
<!-- TOC -->

- [NNArchi Documentation](#nnarchi-documentation)
    - [Task](#task)
    - [Background](#background)
    - [Files and Methods](#files-and-methods)
        - [`NNArchi.py`](#nnarchipy)
            - [get_repo_full_name(*repo_url*)](#get_repo_full_namerepo_url)
            - [extract_architecture_from_python(*repo_full_name*)](#extract_architecture_from_pythonrepo_full_name)
        - [`ast_etree.py`](#ast_etreepy)
            - [rebuild_list(*root*)](#rebuild_listroot)
            - [list_to_tuple(*tuple_list*)](#list_to_tupletuple_list)
            - [rebuild_attr(*root*)](#rebuild_attrroot)
            - [rebuild_lambda_args(*root*)](#rebuild_lambda_argsroot)
            - [rebuild_lambda_expr(*root*)](#rebuild_lambda_exprroot)
            - [get_func_call_paras_kws(*root, has_ext_paras = False,* ***kwarg*)](#get_func_call_paras_kwsroot-has_ext_paras--false-kwarg)
            - [extract_architecture_from_python_ast(*code_str, model_num*)](#extract_architecture_from_python_astcode_str-model_num)
    - [Extraction Steps](#extraction-steps)
    - [Reference](#reference)

<!-- /TOC -->
## Task
Extract neural network architectures (based on ***Keras***[<sup>1</sup>](#refer-anchor)) from source codes (Python)
## Background
Names of public repositories containing projects in the field of neural networks are stored in given datasets (in json files). Each repository contains several models of neural network. What to be extracted is architectures of these models and their parameters. To complete this task, a general method is generated using Python ast odule (Abstract Syntax Trees).
>The ast module helps Python applications to process trees of the Python abstract syntax grammar. The abstract syntax itself might change with each Python release; this module helps to find out programmatically what the current grammar looks like.[<sup>2</sup>](#refer-anchor)  

Sequential model is used in most models based on Keras. For example[<sup>3</sup>](#refer-anchor):
```
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

seq_length = 64

model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=(seq_length, 100)))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
```

With ast module architecture of models could be extracted by parsing abstract syntax trees of source code. Considering complexity of abstract syntax trees and difficulty of parsing, a conversion of abstract syntax trees is needed. Therefore converting abstract syntax trees into XML-etrees is a feasible approach. To realize this approach, two site-packages ***astexport***[<sup>4</sup>](#refer-anchor) and ***json2xml***[<sup>5</sup>](#refer-anchor) are used. First, astexport converts asts into jsons. Then, json2xml converts jsons into xml-etrees. The result of conversion is shown below.  

For example, orginal codes:
```
import os

def func(i):
    i = 2
    return i

a= func(1)
```
<details>  
<summary><b>result of conversion:</b></summary>

```
<?xml version="1.0" ?>
<all>
	<ast_type type="str">Module</ast_type>
	<body type="list">
		<item type="dict">
			<ast_type type="str">Import</ast_type>
			<col_offset type="int">0</col_offset>
			<lineno type="int">1</lineno>
			<names type="list">
				<item type="dict">
					<asname type="null"/>
					<ast_type type="str">alias</ast_type>
					<name type="str">os</name>
				</item>
			</names>
		</item>
		<item type="dict">
			<args type="dict">
				<args type="list">
					<item type="dict">
						<annotation type="null"/>
						<arg type="str">i</arg>
						<ast_type type="str">arg</ast_type>
						<col_offset type="int">11</col_offset>
						<lineno type="int">3</lineno>
					</item>
				</args>
				<ast_type type="str">arguments</ast_type>
				<defaults type="list"/>
				<kw_defaults type="list"/>
				<kwarg type="null"/>
				<kwonlyargs type="list"/>
				<vararg type="null"/>
			</args>
			<ast_type type="str">FunctionDef</ast_type>
			<body type="list">
				<item type="dict">
					<ast_type type="str">Assign</ast_type>
					<col_offset type="int">4</col_offset>
					<lineno type="int">4</lineno>
					<targets type="list">
						<item type="dict">
							<ast_type type="str">Name</ast_type>
							<col_offset type="int">4</col_offset>
							<ctx type="dict">
								<ast_type type="str">Store</ast_type>
							</ctx>
							<id type="str">i</id>
							<lineno type="int">4</lineno>
						</item>
					</targets>
					<value type="dict">
						<ast_type type="str">Num</ast_type>
						<col_offset type="int">8</col_offset>
						<lineno type="int">4</lineno>
						<n type="dict">
							<ast_type type="str">int</ast_type>
							<n type="int">2</n>
							<n_str type="str">2</n_str>
						</n>
					</value>
				</item>
				<item type="dict">
					<ast_type type="str">Return</ast_type>
					<col_offset type="int">4</col_offset>
					<lineno type="int">5</lineno>
					<value type="dict">
						<ast_type type="str">Name</ast_type>
						<col_offset type="int">11</col_offset>
						<ctx type="dict">
							<ast_type type="str">Load</ast_type>
						</ctx>
						<id type="str">i</id>
						<lineno type="int">5</lineno>
					</value>
				</item>
			</body>
			<col_offset type="int">0</col_offset>
			<decorator_list type="list"/>
			<lineno type="int">3</lineno>
			<name type="str">myfunc</name>
			<returns type="null"/>
		</item>
		<item type="dict">
			<ast_type type="str">Assign</ast_type>
			<col_offset type="int">0</col_offset>
			<lineno type="int">7</lineno>
			<targets type="list">
				<item type="dict">
					<ast_type type="str">Name</ast_type>
					<col_offset type="int">0</col_offset>
					<ctx type="dict">
						<ast_type type="str">Store</ast_type>
					</ctx>
					<id type="str">a</id>
					<lineno type="int">7</lineno>
				</item>
			</targets>
			<value type="dict">
				<args type="list">
					<item type="dict">
						<ast_type type="str">Num</ast_type>
						<col_offset type="int">10</col_offset>
						<lineno type="int">7</lineno>
						<n type="dict">
							<ast_type type="str">int</ast_type>
							<n type="int">1</n>
							<n_str type="str">1</n_str>
						</n>
					</item>
				</args>
				<ast_type type="str">Call</ast_type>
				<col_offset type="int">3</col_offset>
				<func type="dict">
					<ast_type type="str">Name</ast_type>
					<col_offset type="int">3</col_offset>
					<ctx type="dict">
						<ast_type type="str">Load</ast_type>
					</ctx>
					<id type="str">myfunc</id>
					<lineno type="int">7</lineno>
				</func>
				<keywords type="list"/>
				<lineno type="int">7</lineno>
			</value>
		</item>
	</body>
</all>
```  

</details>  

As conversion result shown above, functions and their parameters can be extracted by parsing etrees. Therefore, functions like ***Sequential()***, ***add()***, ***compile()*** and their parameters can also be extracted using this approach.

To parse xml etrees, another site-package ***lxml***[<sup>6</sup>](#refer-anchor) is used, which is efficient and supports ***XPath***.

## Files and Methods
### `NNArchi.py`
#### get_repo_full_name(*repo_url*)
Return full name of repository. *`repo_url`* is repository's url stored in dataset.
```
>>> repo_url = 'github.com/francarranza/genre_classification'
>>> get_repo_full_name('repo_url')
'francarranza/genre_classification'
```  

#### extract_architecture_from_python(*repo_full_name*)
Extract architectures of neural network models from .py files in a repository with *`repo_full_name`*.  
```
>>> repo_full_name = 'francarranza/genre_classification'
>>> extract_architecture_from_python(repo_full_name)
{1:{'compile_info': {'loss': 'keras.losses.catego...ssentropy', 'metrics': [...], 'optimizer': 'sgd'}, 'layers': {1: {...}, 2: {...}, 3: {...}, 4: {...}, 5: {...}, 6: {...}, 7: {...}, 8: {...}, 9: {...}, ...}}}
```  
### `ast_etree.py`
#### rebuild_list(*root*)
Rebuild `list` from its etree structure, start position is *`root`*.  
Take *francarranza/genre_classification* as an example. *`root`* corresponds to the tuple in `def genre_classification_baby(input_shape=(128, 130), nb_genres=10):` (in `train.py`, line 55). *code_etree* is previously mentioned conversion result of `train.py`.
```
>>> root = code_etree.xpath('/all/body/item[22]/args/defaults/item[1]')
>>> rebuild_list(root)
[128, 130]
```  
If *`root`* points to a `tuple`, the result should be further converted by using [**list_to_tuple(*tuple_list*)**](#list_to_tuple(tuple_list)).
#### list_to_tuple(*tuple_list*)
Convert a `tuple` in `list` form (*tuple_list*) into `tuple`.  
```
>>> tuple_list = [128, 130]
>>> list_to_tuple(tuple_list)
(128, 130)
```  
#### rebuild_attr(*root*)
Rebuild full attribute name from its etree structure, start position is *`root`*.  
Take *francarranza/genre_classification* as an example. *`root`* corresponds to the value of *`loss`* in `model.compile(loss=keras.losses.categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])` (in `train.py`, line 169-172).
```
>>> root = code_etree.xpath('/all/body/item[33]/value/keywords/item[1]/value')
>>> rebuild_attr(root)
'keras.losses.categorical_crossentropy'
```  
#### rebuild_lambda_args(*root*)
Extract arguments of lambda expressions, start position is *`root`*.
Return a `list` of lambda expression's arguments *`lambda_arg_list`*.
Take *nagyben/CarND-Behavioral-Cloning-P3* as an example. *`root`* corresponds to `model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))` (in `train.py`, line 90).
```
>>>lambda_root = code_etree.xpath('/all/body/item[12]/body/item[2]/value/args/item/args/item')
>>>rebuild_lambda_args(lambda_root)
['x']
```
#### rebuild_lambda_expr(*root*)
Extract lambda expressions, start position is *`root`*.
Return a `str` of lambda expression *`lambda_expr`*.
Take *nagyben/CarND-Behavioral-Cloning-P3* as an example. *`root`* corresponds to `model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))` (in `train.py`, line 90).
```
>>>lambda_root = code_etree.xpath('/all/body/item[12]/body/item[2]/value/args/item/args/item/body')
>>>rebuild_lambda_expr(lambda_root)
'x / 255.0 - 0.5'
```
#### get_func_call_paras_kws(*root, has_ext_paras = False,* ***kwarg*)  
Extract values of parameters and keywords of a function, when it is called, from its etree structure, start position is *`root`*. Return a `list` of parameters' values *`func_call_paras_list`* and a `dictionary` of keywords' values *`func_call_kws_dict`*.  
Take *francarranza/genre_classification* as an example. *`root`* corresponds to `model = genre_classification_baby(input_shape=(128, 388), nb_genres=5)` (in `train.py`, line 156-157). 
```
>>> func_call_root = code_etree.xpath('/all/body/item[29]/value')
>>> get_func_call_paras_kws(func_call_root, has_ext_paras = True)
[] #func_call_paras_list
{'input_shape': (128, 388), 'nb_genres': '5'} #func_call_kws_dict
```  
#### extract_architecture_from_python_ast(*code_str, model_num*)
Extract architectures of neural networks from python ast etree of source code. *`code_str`* is source code in `string` form and *`model_num`* is the number of models already found in the repository.
Take *francarranza/genre_classification* as an example. *`code_str`* is `train.py` in `string` form.
```
model_num = 0
extract_architecture_from_python_ast(code_str, model_num)
{1:{'compile_info': {'loss': 'keras.losses.catego...ssentropy', 'metrics': [...], 'optimizer': 'sgd'}, 'layers': {1: {...}, 2: {...}, 3: {...}, 4: {...}, 5: {...}, 6: {...}, 7: {...}, 8: {...}, 9: {...}, ...}}}
```  
## Extraction Steps
1. Load data set and collect names of repositories. Complete full url of repository using [`get_repo_full_name(repo_url)`](#get_repo_full_namerepo_url)  
2. Using Github API to search .py files, which imports `keras`, in a repository. Load these .py files as `string`.  
3. Convert source code in `string` form into `ast` in form of XML-etree.  
4. Parse XML-etree to search `Sequential()`. Once found, extract the name of sequential model with target of assignment and check ancestor nodes of current nodes using `XPath` method `xpath('ancestor::item[ast_type = "FunctionDef"]')` to find definition of custom function.  
5. If definition of custom fuction exists, which means model is built inside a function, extract default values of parameters and keywords. Model-root is beginning of function.  
6. Check if this custom function is called. If called, extract values of parameters and keywords. Replace the default values with them.  
7. If definition of custom fuction doesn't exist, which means model is directly built, model-root is beginning of whole source code.  
8. Parse etree from model-root. If expression is found while its `id` equals name of sequential model (in Step 4) and its `attr` equals `add`, which means keras method `add()` is found and a new layer is added to model, extract its name and its values of parameters and keywords.  
9. If expression's `id` equals `compile`, which means keras method `compile()` is found and the model is compiled, extract its values of keywords.  
10. After parsing this part of etree, check if model is compiled. If not, parse whole etree to find `compile()` corresponding to this model and extract its values of keywords. If nothing is found, it means model is not compiled.  
11. Return architecture of model by integrating information of layes and `compile` from previous steps 8-10.

<div id="refer-anchor"></div>

## Reference
[1] [Keras: The Python Deep Learning library](https://keras.io/)  
[2] [Python AST Module — Abstract Syntax Trees](https://docs.python.org/3.7/library/ast.html)  
[3] [Getting started with the Keras Sequential model](https://keras.io/getting-started/sequential-model-guide/)  
[4] [astexport · PyPI](https://pypi.org/project/astexport/)  
[5] [json2xml · PyPI](https://pypi.org/project/json2xml/)  
[6] [lxml - Processing XML and HTML with Python](https://lxml.de/index.html)  

## License

This project is licensed under the Creative Commons BY 4.0 license - see [here](https://creativecommons.org/licenses/by/4.0/) for details

