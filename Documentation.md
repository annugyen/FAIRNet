# NNArchi Documentation
## Task
Extract neural network architectures (based on ***Keras***[<sup>1</sup>](#refer-anchor)) from source codes (Python)
## Background
Names of public repositories containing projects in the field of neural networks are stored in given datasets (in json files). Each repository contains several models of neural network. What to be extracted is architectures of these models and their parameters. To complete this task, a general method is generated using Python ast odule (Abstract Syntax Trees).
>The ast module helps Python applications to process trees of the Python abstract syntax grammar. The abstract syntax itself might change with each Python release; this module helps to find out programmatically what the current grammar looks like.[<sup>2</sup>](#refer-anchor)  

Sequential model is used in most models based on Keras. For example[<sup>3</sup>](#refer-anchor)  :
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
result of conversion:
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
As shown above, functions and their parameters can be extracted by parsing etrees. Therefore, functions like ***Sequential()***, ***add()***, ***compile()*** and their parameters can also be extracted using this approach.

To prase xml etrees, another site-package ***lxml***[<sup>6</sup>](#refer-anchor) is used.

<div id="refer-anchor"></div>

## Reference
[1] [Keras: The Python Deep Learning library](https://keras.io/)  
[2] [Python AST Module — Abstract Syntax Trees](https://docs.python.org/3.7/library/ast.html)  
[3] [Getting started with the Keras Sequential model](https://keras.io/getting-started/sequential-model-guide/)  
[4] [astexport · PyPI](https://pypi.org/project/astexport/)  
[5] [json2xml · PyPI](https://pypi.org/project/json2xml/)  
[6] [lxml - Processing XML and HTML with Python](https://lxml.de/index.html)  