import json
import os
import random
import re
import string
import sys
from contextlib import closing
from urllib import request

import pandas as pd
import tensorflow as tf
from keras.backend import clear_session
from keras.engine.saving import load_model

from githubtokens import Token_list

neuronStoplist = ['add', 'dot', 'subtract', 'multiply', 'average', 'maximum', 'minimum', 'concatenate']

ROOT_DIR = './'

def get_activation_function(function_name):
    """
    Transform keras activation function name into full activation function name
    :param function_name: Keras activation function name
    :return: Full activation function name
    """

    translation_dict = {
        'relu': 'Rectified Linear Unit',
        'linear': 'Linear',
        'elu': 'Exponential Linear Unit',
        'exponential': 'Exponential',
        'selu': 'Scaled Exponential Linear Unit',
        'tanh': 'Hyperbolic Tangent',
        'sigmoid': 'Sigmoid',
        'hard_sigmoid': 'Hard Sigmoid',
        'softmax': 'Softmax',
        'softplus': 'Softplus',
        'softsign': 'Softsign',
    }

    return_name = translation_dict.get(function_name, function_name.capitalize())

    return return_name

def extract_architecture_from_h5(model_url):
    """
    Extracts model architecture from h5 file.

    :param model_url: Link to h5 file on GitHub
    :return:
    """

    #sys.stdout.write('\rExtract model architecture from h5 file ...')
    #sys.stdout.flush()

    model_url_raw = model_url.replace('blob', 'raw')
    # print(model_url_raw)
    temp_file_name = 'temp_file_' + ''.join(random.choices(string.ascii_lowercase + string.digits, k=3))
    temp_file_path = os.path.join(ROOT_DIR, 'h5example/' + temp_file_name + '.h5')
    Token_idx = random.randint(0, len(Token_list) - 1)
    headers = {'Authorization':'token ' + Token_list[Token_idx]}
    h5_request = request.Request(model_url_raw, headers = headers)
    h5_response = None
    h5_read = None
    try:
        with closing(request.urlopen(h5_request)) as h5_response:
            h5_read = h5_response.read()
    except Exception as e:
        print('Download Error')
    else:
        with open(temp_file_path, 'wb') as f:
            f.write(h5_read)
        f.close()
        h5_response = None
        h5_read = None

    #path, header = request.urlretrieve(model_url_raw, temp_file_path)

    # print(path)
    #sys.stdout.write('\rLoad model from path ...')
    #sys.stdout.flush()

    # Set variable to default values
    extracted_architecture = False
    loss_function = None
    optimizer = None
    layers = None
    metrics = None

    # print('\n\n\nModel:\n')
    try:
        # Try loading model architecture from file
        model = load_model(temp_file_path)
    except ValueError as ve:
        print('Value error occurred. %s' % ve.args)
    except SystemError as se:
        print('System error occurred. %s' % se.args)
    except Exception as e:
        print('Unknown exception occurred. %s' % e.args)
    else:
        # In case model could be successfully loaded
        #sys.stdout.write('\rLoading model successful ...')
        #sys.stdout.flush()

        extracted_architecture = True
        layers = dict()

        # Iterate through layers
        for idx, layer in enumerate(model.layers):
            # print(layer.get_config())
            # print(str(layer))
            layer_name = layer.name
            layer_type = re.search(r'<keras\..+\..+\.(.*?)\s.*', str(layer)).group(1)
            layer_sequence = idx
            activation_function = None
            nr_neurons = None
            try:
                # Extract number of neurons of current layer
                if layer.name not in set(neuronStoplist):
                    nr_neurons = int(layer.input_shape[1])  # Default number of neurons
                    if layer_type == 'Dropout':
                        nr_neurons = int(layer.input_shape[1] * (1 - layer.rate))
                    elif len(layer.input_shape) == 3:
                        nr_neurons = int(layer.input_shape[1] * layer.input_shape[2])
                    elif len(layer.input_shape) == 4:
                        nr_neurons = int(layer.input_shape[1] * layer.input_shape[2] * layer.input_shape[3])
            except Exception as e:
                nr_neurons = '?'  # Value assigned if extraction fails

            try:
                # Extract activation function
                activation_function = get_activation_function(layer.output.op.op_def.name.lower())
            except Exception as e:
                pass

            # Build layer as dictionary entry
            layers[str(layer_sequence)] = {'layer_name': layer_name, 'layer_type': layer_type,
                                           'nr_neurons': nr_neurons, 'activation_function': activation_function}

        try:
            loss_function = model.loss.lower()
        except AttributeError as ae:
            # print('Has no loss function.')
            pass

        try:
            optimizer = str(model.optimizer).split()[0].replace('<keras.optimizers.', '')
        except AttributeError as ae:
            # print('Has no optimizer function.')
            pass
        
        try:
            metrics = model.metrics_names
        except AttributeError as ae:
            pass

    finally:
        os.remove(temp_file_path)  # Delete temporary file
        # print(layers)
        clear_session()
        tf.compat.v1.reset_default_graph()
        return extracted_architecture, loss_function, optimizer, metrics, layers

if __name__ == '__main__':
    #model_url = 'https://github.com/nagyben/CarND-Behavioral-Cloning-P3/blob/f044a3a1eff5b30171b763eb2eda0b2dba19469e/model.h5' #index 1 fixed
    #model_url = 'https://github.com/XintongHao/Self-Driving-Car-Behavioral-Cloning/raw/master/model.h5' #index 10 fixed
    #model_url = 'https://github.com/UlucFVardar/Painting-Matcher-Project/raw/f00948d8875dfcd9be98edfce446ab9991890bc0/CNN%20implementation/net.h5' #index 100 fixed
    #model_url = 'https://github.com/gayaviswan/Udacity-Behavioural-Cloning/raw/1733a246a3ca90c7697f4ae982307337b9b412b2/model.h5'
    #model_url = 'https://github.com/sendilk/ml-class/blob/850d4d1ecbd18c37a401baefc0deb6b3d06dc326/keras-autoencoder/auto-denoise.h5' #index 43, 14 layers
    #model_url = 'https://github.com/sendilk/ml-class/blob/107427b4829f92e166577d87ef3e3d0b5f8eea21/two-layer.h5' #index 43, 5 layers
    #model_url = 'https://github.com/sanketx/MP3vec/blob/74622144e96767dab126e256d208ce33b9882d3b/mp3vec/mp3_model.h5' #index 99, No training configuration found in save file
    #extracted_architecture, loss_function, optimizer, layers = extract_architecture_from_h5(model_url)
    
    data_path = './filtered_data.json'
    with open(data_path, 'r') as file:
        data = json.load(file)
    file.close()
    df = pd.DataFrame(data)
    for idx in range(df.index.size):
        if df.loc[str(idx), 'h5_data']['extracted_architecture']:
            model_url = df.loc[str(idx), 'h5_files_links'][0]
            extracted_architecture, loss_function, optimizer, layers = extract_architecture_from_h5(model_url)
            if (not df.loc[str(idx), 'h5_data']['loss_function']) or (not df.loc[str(idx), 'h5_data']['optimizer']):
                if loss_function or optimizer:
                    print('%d: h5_data can be updated' % idx)
                else:
                    print('%d: h5_data still has problems' % idx)
        if idx == 100:
            break
    pass
