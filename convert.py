import datetime
import json

from rdflib import DOAP, OWL, RDF, RDFS, Graph, Literal, Namespace, URIRef, BNode

substitution_dict = {'rectified_linear_unit': 'relu',
                     'exponential_linear_unit': 'elu',
                     'scaled_exponential_linear_unit': 'selu',
                     'hyperbolic_tangent': 'tanh'}

def convert_owl(data_json, result_json, owl_path):

    nno_url = 'https://w3id.org/nno/ontology#'
    base_url = 'https://w3id.org/nno/data#'
    nno = Namespace(nno_url)
    dc = Namespace('http://purl.org/dc/terms/')
    vs = Namespace('http://www.w3.org/2003/06/sw-vocab-status/ns#')
    cc = Namespace('http://creativecommons.org/ns#')
    xmls = Namespace('http://www.w3.org/2001/XMLSchema#')
    
    g = Graph()
    g.parse('http://people.aifb.kit.edu/ns1888/nno/nno.owl', format='xml')

    activation_functions = g.subjects(RDF.type, URIRef(nno_url + 'Activation_Function'))
    activation_functions_set = set()
    for func in activation_functions:
        activation_functions_set.add(str(func))

    loss_funtions_regr = g.subjects(RDF.type, URIRef(nno_url + 'Regressive_Loss'))
    loss_funtions_class = g.subjects(RDF.type, URIRef(nno_url + 'Classification_Loss'))
    loss_functions_set = set()
    for loss in loss_funtions_regr:
        loss_functions_set.add(str(loss))
    for loss in loss_funtions_class:
        loss_functions_set.add(str(loss))
    
    optimizers = g.subjects(RDF.type, URIRef(nno_url + 'Optimizer'))
    optmizers_set = set()
    for opt in optimizers:
        optmizers_set.add(str(opt))

    #g = Graph()
    g.bind('nno', nno)
    g.bind('dc', dc)
    g.bind('owl', OWL)
    g.bind('vs', vs)
    g.bind('cc', cc)
    g.bind('xmls', xmls)
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
    g.add((nno.hassuggestedUse, RDFS.domain, nno.Neural_Network))
    g.add((nno.hassuggestedUse, RDFS.range, xmls.string))
    g.add((nno.hassuggestedUse, RDFS.label, Literal('has suggested use')))
    g.add((nno.hassuggestedUse, RDFS.comment, Literal('Suggested primary intended use (domain) for which the Neural Network was trained for.')))
    
    g.add((URIRef(nno_url + 'hassuggestedType'), RDF.type, OWL.DatatypeProperty))
    g.add((nno.hassuggestedType, RDFS.domain, nno.Neural_Network))
    g.add((nno.hassuggestedType, RDFS.range, xmls.string))
    g.add((nno.hassuggestedType, RDFS.label, Literal('has suggested type')))
    g.add((nno.hassuggestedType, RDFS.comment, Literal('Suggested Neural Network type based on readme information.')))

    g.add((URIRef(nno_url + 'Model'), RDF.type, OWL.Class))
    g.add((nno.Model, RDFS.label, Literal('Model')))
    g.add((nno.Model, RDFS.comment, Literal('Model of the repository.')))

    g.add((URIRef(nno_url + 'hasModel'), RDF.type, OWL.ObjectProperty))
    g.add((nno.hasModel, RDFS.domain, nno.Neural_Network))
    g.add((nno.hasModel, RDFS.range, nno.Model))
    g.add((nno.hasModel, RDFS.label, Literal('has model')))
    g.add((nno.hasModel, RDFS.comment, Literal('Model of the repository.')))

    g.add((URIRef(nno_url + 'hasModelSequence'), RDF.type, OWL.DatatypeProperty))
    g.add((nno.hasModelSequence, RDFS.domain, nno.Model))
    g.add((nno.hasModelSequence, RDFS.range, xmls.int))
    g.add((nno.hasModelSequence, RDFS.label, Literal('has model sequence')))
    g.add((nno.hasModelSequence, RDFS.comment, Literal('Specifies the sequence of the models, starts at 1')))
    
    g.remove((nno.hasLayer, RDFS.domain, nno.Neural_Network))
    g.add((nno.hasLayer, RDFS.domain, nno.Model))

    g.remove((nno.hasLossFunction, RDFS.domain, nno.Neural_Network))
    g.add((nno.hasLossFunction, RDFS.domain, nno.Model))

    g.remove((nno.hasOptimizer, RDFS.domain, nno.Neural_Network))
    g.add((nno.hasOptimizer, RDFS.domain, nno.Model))

    g.remove((nno.hasMetric, RDFS.domain, nno.Neural_Network))
    g.add((nno.hasMetric, RDFS.domain, nno.Model))
    
    for key in result_json:
        idx = int(key)
        result = result_json[key]
        data = data_json[idx]
        repo_full_name = result['repo_full_name']
        owner = 'https://github.com/' + data['repo_owner']
        
        repo = URIRef(base_url + repo_full_name)
        g.add((repo, RDF.type, nno.Neural_Network))
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

        if data.get('license') and data['license'].get('url'):
            repo_license = URIRef(data['license']['url'])
            g.add((repo, dc.license, repo_license))
        
        if data.get('reference_list'):
            for ref in data['reference_list']:
                g.add((repo, dc.references, URIRef(ref)))
        
        if data.get('see_also_links'):
            for see in data['see_also_links']:
                g.add((repo, RDFS.seeAlso, URIRef(see)))
        
        if result.get('models') and result['models'] != 'Error':
            models = result['models']
            for model_idx in models:
                model = models[model_idx]
                model_name = 'model_' + model_idx
                model_URI = URIRef(base_url + repo_full_name + '_' + model_name)
                g.add((model_URI, RDFS.label, Literal(model_name)))
                g.add((model_URI, nno.hasModelSequence, Literal(model_idx)))
                g.add((model_URI, RDF.type, nno.Model))
                
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
                    layer_type = layer['layer_type']
                    layer_name = layer['layer_name']
                    layer_URI = URIRef(base_url + repo_full_name + '_' + model_name + '_' + layer_name)
                    layer_type_URI = URIRef(nno_url + layer_type)

                    #g.add((model_URI, nno.hasLayer, layer_URI)) #connect layer to model
                    g.add((layer_URI, RDFS.label, Literal(layer_name)))
                    g.add((layer_URI, nno.hasLayerSequence, Literal(int(layer_idx) + 1)))
                    g.add((layer_URI, RDF.type, layer_type_URI))                  
                    
                    neurons = layer.get('nr_neurons')
                    g.add((layer_URI, nno.hasNeurons, Literal(neurons)))

                    if layer.get('activation_function'):
                        activation_function_full_name = layer['activation_function']
                    else:
                        activation_function_full_name = 'Linear'
                    activation_function = activation_function_full_name.replace(' ', '_').lower()

                    if activation_function in substitution_dict:
                        activation_function = substitution_dict[activation_function]
                    
                    activation_function_URI = URIRef(nno_url + activation_function)
                    if str(activation_function_URI) in activation_functions_set:
                        g.add((layer_URI, nno.hasActivationFunction, activation_function_URI))
                        pass
                    else:
                        print('activation %s is not in set.' % activation_function_URI)
                        
                    g.add((model_URI, nno.hasLayer, layer_URI)) #connect layer to model
                    
                if model.get('compile_info'):
                    compile_info = model['compile_info']
                    if compile_info.get('loss'):
                        loss_full_name = compile_info['loss']
                        loss_func = loss_full_name.replace(' ', '_').lower()
                    
                        if loss_func == 'mse':
                            loss_func = 'mean_squared_error'
                        
                        loss_func_URI = URIRef(nno_url + loss_func)

                        if str(loss_func_URI) in loss_functions_set:
                            g.add((model_URI, nno.hasLossFunction, loss_func_URI))
                        else:
                            print('loss %s is not in set' % loss_func_URI)

                    if compile_info.get('optimizer'):
                        optimizer_full_name = compile_info['optimizer']
                        optimizer = optimizer_full_name.replace(' ', '_').lower()
                        optimizer_URI = URIRef(nno_url + optimizer)

                        if str(optimizer_URI) in optmizers_set:
                            g.add((model_URI, nno.hasOptimizer, optimizer_URI))
                        else:
                            print('optimizer %s is not in set.' % optimizer_URI)

                    if compile_info.get('metrics'):
                        g.add((model_URI, nno.hasMetric, Literal(str(compile_info['metrics']))))
                

    g.serialize(owl_path, format = 'pretty-xml')

if __name__ == '__main__':
    data_path = './data.json'
    with open(data_path, 'r') as f:
        data_json = json.load(f)
    f.close()

    result_path = './result_data_h5.json'
    with open(result_path, 'r') as f:
        result_json = json.load(f)
    f.close()

    owl_path = './result_data_h5.owl'

    convert_owl(data_json, result_json, owl_path)
