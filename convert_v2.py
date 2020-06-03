import datetime
import json
from copy import deepcopy

from rdflib import DOAP, OWL, RDF, RDFS, Graph, Literal, Namespace, URIRef, BNode
from werkzeug.urls import url_fix

from check_overlap import layer_dict, trans_opti, trans_loss

substitution_dict = {'rectified_linear_unit': 'relu',
                     'exponential_linear_unit': 'elu',
                     'scaled_exponential_linear_unit': 'selu',
                     'hyperbolic_tangent': 'tanh'}

def gather_layer_keywords(layer):
    keywords = deepcopy(layer)
    if 'parameters' in keywords:
        del keywords['parameters']
    if 'layer_type' in keywords:
        del keywords['layer_type']
    return keywords

def convert_owl(data_json, result_json, owl_path):

    nno_url = 'https://w3id.org/nno/ontology#'
    base_url = 'https://w3id.org/nno/data#'
    nno = Namespace(nno_url)
    dc = Namespace('http://purl.org/dc/terms/')
    vs = Namespace('http://www.w3.org/2003/06/sw-vocab-status/ns#')
    cc = Namespace('http://creativecommons.org/ns#')
    xmls = Namespace('http://www.w3.org/2001/XMLSchema#')
    
    g = Graph()
    #g.parse('http://people.aifb.kit.edu/ns1888/nno/nno.owl', format='xml')
    #g.parse('./nno.owl', format='xml')
    g.parse('http://people.aifb.kit.edu/ns1888/nno/nno.ttl', format='turtle')

    activation_functions = g.subjects(RDF.type, URIRef(nno_url + 'ActivationFunction'))
    activation_functions_set = set()
    for func in activation_functions:
        activation_functions_set.add(str(func))

    g.add((URIRef(nno_url + 'huberloss'), RDF.type, OWL.NamedIndividual))
    g.add((nno.huberloss, RDF.type, nno.RegressiveLoss))
    g.add((nno.huberloss, RDFS.label, Literal('Huber Loss')))

    loss_funtions_regr = g.subjects(RDF.type, URIRef(nno_url + 'RegressiveLoss'))
    loss_funtions_class = g.subjects(RDF.type, URIRef(nno_url + 'ClassificationLoss'))
    loss_functions_set = set()
    for loss in loss_funtions_regr:
        loss_functions_set.add(str(loss))
    for loss in loss_funtions_class:
        loss_functions_set.add(str(loss))
    
    optimizers = g.subjects(RDF.type, URIRef(nno_url + 'Optimizer'))
    optmizers_set = set()
    for opt in optimizers:
        optmizers_set.add(str(opt))

    layer_class_set = set()
    all_layer_class = g.subjects(RDFS.subClassOf, nno.Layer)
    for layer_class in all_layer_class:
        layer_class_set.add(str(layer_class))
    
    layer_type_set = set()
    for layer_class in layer_class_set:
        layer_types = g.subjects(RDFS.subClassOf, URIRef(layer_class))
        for l in layer_types:
            layer_type_set.add(str(l))
    
    '''#test
    cnn_layer_type = set()
    cnn_layers = g.subjects(RDFS.subClassOf, nno.ConvolutionalLayer)
    for l in cnn_layers:
        cnn_layer_type.add(str(l))
    
    rnn_layer_type = set()
    rnn_layers = g.subjects(RDFS.subClassOf, nno.RecurrentLayer)
    for l in rnn_layers:
        rnn_layer_type.add(str(l))
    '''

    #g = Graph()
    g.bind('nno', nno)
    g.bind('dc', dc)
    g.bind('owl', OWL)
    g.bind('vs', vs)
    g.bind('cc', cc)
    g.bind('xsd', xmls)
    g.bind('doap', DOAP)

    tmp = URIRef('https://w3id.org/nno/data')
    g.add((tmp, dc.publisher, URIRef('http://www.aifb.kit.edu/web/Web_Science')))
    g.add((tmp, dc.title, Literal('FAIRnets Dataset')))
    g.add((tmp, dc.description, Literal('This is the FAIRnets dataset. It contains Information about publicly available Neural Networks.')))

    date_today = datetime.date.today().isoformat()
    g.add((tmp, dc.issued, Literal(date_today)))
    g.add((tmp, RDFS.label, Literal('FAIRnets Dataset')))
    g.add((tmp, vs.term_status, Literal('stable')))
    g.add((tmp, cc.licence, URIRef('https://creativecommons.org/licenses/by-nc-sa/4.0/')))

    g.add((URIRef(nno_url + 'hassuggestedUse'), RDF.type, OWL.DatatypeProperty))
    g.add((nno.hassuggestedUse, RDFS.domain, nno.NeuralNetwork))
    g.add((nno.hassuggestedUse, RDFS.range, xmls.string))
    g.add((nno.hassuggestedUse, RDFS.label, Literal('has suggested use')))
    g.add((nno.hassuggestedUse, RDFS.comment, Literal('Suggested primary intended use (domain) for which the Neural Network was trained for.')))
    
    g.add((URIRef(nno_url + 'hassuggestedType'), RDF.type, OWL.DatatypeProperty))
    g.add((nno.hassuggestedType, RDFS.domain, nno.NeuralNetwork))
    g.add((nno.hassuggestedType, RDFS.range, xmls.string))
    g.add((nno.hassuggestedType, RDFS.label, Literal('has suggested type')))
    g.add((nno.hassuggestedType, RDFS.comment, Literal('Suggested Neural Network type based on readme information.')))

    '''
    g.add((URIRef(nno_url + 'CustomLayer'), RDF.type, OWL.Class))
    g.add((nno.CustomLayer, RDFS.label, Literal('Custom Layer')))
    g.add((nno.CustomLayer, RDFS.comment, Literal('Custom layer defined by user')))
    g.add((nno.CustomLayer, RDFS.subClassOf, nno.Layer))

    g.add((URIRef(nno_url + 'customoptimizer'), RDF.type, OWL.NamedIndividual))
    g.add((nno.customoptimizer, RDF.type, nno.Optimizer))
    g.add((nno.customoptimizer, RDFS.label, Literal('Custom Optimizer')))

    g.add((URIRef(nno_url + 'customloss'), RDF.type, OWL.NamedIndividual))
    g.add((nno.customloss, RDF.type, nno.LossFunction))
    g.add((nno.customloss, RDFS.label, Literal('Custom Loss Function')))

    g.add((URIRef(nno_url + 'hasLayerParameters'), RDF.type, OWL.DatatypeProperty))
    g.add((nno.hasLayerParameters, RDFS.domain, nno.Layer))
    g.add((nno.hasLayerParameters, RDFS.label, Literal('has layer parameters')))
    g.add((nno.hasLayerParameters, RDFS.comment, Literal('Parameters of a layer')))

    g.add((URIRef(nno_url + 'hasLayerKeywords'), RDF.type, OWL.DatatypeProperty))
    g.add((nno.hasLayerKeywords, RDFS.domain, nno.Layer))
    g.add((nno.hasLayerKeywords, RDFS.label, Literal('has layer keywords')))
    g.add((nno.hasLayerKeywords, RDFS.comment, Literal('Keywords of a layer')))

    g.add((URIRef(nno_url + 'hasBaseModel'), RDF.type, OWL.DatatypeProperty))
    g.add((nno.hasBaseModel, RDFS.domain, nno.Model))
    g.add((nno.hasBaseModel, RDFS.label, Literal('has base model')))
    g.add((nno.hasBaseModel, RDFS.comment, Literal('Base model from keras application')))

    g.add((URIRef(nno_url + 'BaseModel'), RDF.type, OWL.Class))
    g.add((nno.BaseModel, RDFS.label, Literal('Base Model')))
    g.add((nno.BaseModel, RDFS.comment, Literal('Base model from keras application')))
    g.add((nno.BaseModel, RDFS.subClassOf, nno.Layer))

    g.add((URIRef(nno_url + 'hasBaseModelKeywords'), RDF.type, OWL.DatatypeProperty))
    g.add((nno.hasBaseModelKeywords, RDFS.domain, nno.BaseModel))
    g.add((nno.hasBaseModelKeywords, RDFS.label, Literal('has base model keywords')))
    g.add((nno.hasBaseModelKeywords, RDFS.comment, Literal('Keywords of base model')))
    '''

    
    for key in result_json:
        idx = int(key)
        result = result_json[key]
        data = data_json[idx]

        if result.get('models') and isinstance(result['models'], dict) and len(result['models']) > 0:

            repo_full_name = result['repo_full_name']
            
            owner = 'https://github.com/' + data['repo_owner']
            
            repo = URIRef(base_url + repo_full_name)

            '''
            if 'nn_type' in data:
                nn_type = data['nn_type']
            elif 'suggested_type' in data:
                nn_type = data['suggested_type']
            else:
                nn_type = ['nn']

            if 'recurrent_type' in nn_type:
                g.add((repo, RDF.type, nno.RecurrentNeuralNetwork))
            elif 'conv_type' in nn_type:
                g.add((repo, RDF.type, nno.ConvolutionalNeuralNetwork))
            elif 'feed_forward_type' in nn_type:
                g.add((repo, RDF.type, nno.FeedForwardNeuralNetwork))
            else:
                g.add((repo, RDF.type, nno.NeuralNetwork))
            '''
            g.add((repo, RDF.type, nno.NeuralNetwork))

            g.add((repo, RDFS.label, Literal(repo_full_name)))

            if data.get('repo_desc'):
                g.add((repo, dc.description, Literal(data['repo_desc'])))
            
            if data.get('readme_text'):
                g.add((repo, dc.description, Literal(data['readme_text'])))

            if owner.startswith('http'):
                g.add((repo, dc.creator, URIRef(owner)))

            if data.get('repo_url').startswith('http'):
                g.add((repo, nno.hasRepositoryLink, URIRef(data['repo_url'])))

            if data.get('repo_last_mod'):
                g.add((repo, dc.modified, Literal(data['repo_last_mod'])))
            
            if data.get('repo_created_at'):
                g.add((repo, dc.created, Literal(data['repo_created_at'])))

            g.add((repo, dc.publisher, URIRef('https://github.com')))

            if data.get('repo_tags'):
                for category_tag in data['repo_tags']:
                    g.add((repo, DOAP.category, Literal(category_tag)))
            
            if data.get('application'):
                for nn_application in data['application']:
                    g.add((repo, nno.hasintendedUse, Literal(nn_application)))
            
            if data.get('repo_watch'):
                g.add((repo, nno.stars, Literal(int(data['repo_watch']))))

            if data.get('license') and data['license'].get('key'):
                repo_license = URIRef('https://choosealicense.com/licenses/' + data['license']['key'])
                g.add((repo, dc.license, repo_license))
            
            if data.get('reference_list'):
                for ref in data['reference_list']:
                    g.add((repo, dc.references, URIRef(url_fix(ref))))
            
            if data.get('see_also_links'):
                for see in data['see_also_links']:
                    g.add((repo, RDFS.seeAlso, URIRef(url_fix(see))))
            
            #if result.get('models') and isinstance(result['models'], dict):
            models = result['models']
            for model_idx in models:
                model = models[model_idx]
                model_name = 'model_' + model_idx
                model_URI = URIRef(base_url + repo_full_name + '_' + model_name)
                g.add((model_URI, RDFS.label, Literal(model_name)))
                g.add((model_URI, nno.hasModelSequence, Literal(int(model_idx))))
                g.add((model_URI, RDF.type, nno.Model))

                if 'base_model' in model:
                    base_model_name = model['base_model']['name']
                    base_model_URI = URIRef(base_url + repo_full_name + '_' + model_name + '_' + base_model_name)
                    g.add((base_model_URI, RDF.type, nno.BaseModel))
                    g.add((base_model_URI, RDFS.label, Literal(base_model_name)))
                    base_model_kw = deepcopy(model['base_model']) 
                    del base_model_kw['name']
                    del base_model_kw['parameters']
                    g.add((base_model_URI, nno.hasBaseModelKeywords, Literal(str(base_model_kw))))
                    g.add((model_URI, nno.hasBaseModel, base_model_URI))

                if 'model_type' in model:
                    if 'cnn' in model['model_type'] or 'rnn' in model['model_type'] or 'fnn' in model['model_type']:
                        g.add((model_URI, nno.hasModelType, Literal(str(model['model_type']))))
                
                g.add((repo, nno.hasModel, model_URI)) #connect model to repo

                '''
                #test
                test_node = BNode(URIRef(base_url + repo_full_name + '_' + model_name + '_test'))
                g.add((test_node, RDF.type, nno.Dense))
                g.add((test_node, RDFS.label, Literal('test_layer')))
                g.add((model_URI, nno.hasLayer, test_node))
                '''

                layers = model['layers']
                for layer_idx in layers:
                    layer = layers[layer_idx]
                    layer_type = layer['layer_type'] if layer['layer_type'] else 'Unknown'
                    layer_type = layer_dict.get(layer_type, layer_type)
                    layer_type = layer_type.replace(' ', '_')
                    layer_name = layer_type.lower() + '_' + layer_idx
                    layer_URI = URIRef(base_url + repo_full_name + '_' + model_name + '_' + layer_name)
                    layer_type_URI = URIRef(nno_url + layer_type) if (nno_url + layer_type) in layer_type_set else nno.CustomLayer

                    #g.add((model_URI, nno.hasLayer, layer_URI)) #connect layer to model
                    g.add((layer_URI, RDFS.label, Literal(layer_name)))
                    g.add((layer_URI, nno.hasLayerSequence, Literal(int(layer_idx))))
                    g.add((layer_URI, RDF.type, layer_type_URI))

                    if layer.get('parameters'):
                        layer_parameters = layer['parameters']
                        g.add((layer_URI, nno.hasLayerParameters, Literal(str(layer_parameters))))

                    layer_keywords = gather_layer_keywords(layer)
                    if len(layer_keywords) > 0:
                        g.add((layer_URI, nno.hasLayerKeywords, Literal(str(layer_keywords))))
                        
                    g.add((model_URI, nno.hasLayer, layer_URI)) #connect layer to model

                
                if model.get('compile_info'):
                    compile_info = model['compile_info']
                    if compile_info.get('loss'):
                        loss_full_name = compile_info['loss']
                        loss_func = trans_loss(loss_full_name)
                        loss_func = str(loss_func).replace('[', '').replace(']', '')
                        if loss_func:
                            if ((nno_url + loss_func) in loss_functions_set):
                                loss_func_URI = URIRef(nno_url + loss_func) 
                                g.add((model_URI, nno.hasLossFunction, loss_func_URI))
                            else:
                                loss_func_URI = URIRef(base_url + repo_full_name + '_' + model_name + '_customloss')
                                g.add((model_URI, nno.hasLossFunction, loss_func_URI))
                                g.add((loss_func_URI, RDF.type, nno.customloss))
                                g.add((loss_func_URI, RDFS.label, Literal(str(loss_full_name))))

                    if compile_info.get('optimizer'):
                        optimizer_full_name = compile_info['optimizer']
                        optimizer = trans_opti(optimizer_full_name).lower()
                        if optimizer:
                            if (nno_url + optimizer) in optmizers_set:
                                optimizer_URI = URIRef(nno_url + optimizer)
                                g.add((model_URI, nno.hasOptimizer, optimizer_URI))
                            else:
                                optimizer_URI = URIRef(base_url + repo_full_name + '_' + model_name + '_customoptimizer')
                                g.add((model_URI, nno.hasOptimizer, optimizer_URI))
                                g.add((optimizer_URI, RDF.type, nno.customoptimizer))
                                g.add((optimizer_URI, RDFS.label, Literal(str(optimizer_full_name))))

                    if compile_info.get('metrics'):
                        g.add((model_URI, nno.hasMetric, Literal(str(compile_info['metrics']))))
        '''#test
        if idx == 99:
            break 
        '''
                

    g.serialize(owl_path, format = 'turtle')

if __name__ == '__main__':
    data_path = './data.json'
    with open(data_path, 'r') as f:
        data_json = json.load(f)
    f.close()

    result_path = './result_data_v6.json'
    with open(result_path, 'r') as f:
        result_json = json.load(f)
    f.close()

    owl_path = './result_data_v7.ttl'

    convert_owl(data_json, result_json, owl_path)

    a = 1
