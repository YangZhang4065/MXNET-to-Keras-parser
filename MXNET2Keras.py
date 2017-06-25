import json

import numpy as np
import mxnet as mx

from keras.models import Model
from keras.layers import Input, Dense, Conv2D
from keras.layers import Activation,BatchNormalization
from keras.layers import Add,Concatenate
from keras.layers import AveragePooling2D,MaxPooling2D,Dropout
from keras.layers import Lambda,Flatten
from keras import backend as K

def str_to_bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return False
    else:
         raise ValueError

def MXNET2Keras(prefix,input_shape,verbose=1,iteration=0):
    model_loaded,loaded_params,loaded_aux_params = mx.model.load_checkpoint(prefix, iteration)
    
    #parse parameters
    params = {}
    for layer_param_key in loaded_params:
        params[layer_param_key] = loaded_params[layer_param_key].asnumpy()
    
    #parse network connection
    network_layers=list()
    nodes = json.loads(model_loaded.tojson())['nodes']
    for i_node in nodes:
        if str(i_node['op']) != 'null':
            i_node['input_layer_name']=list()
            
            for input_n in i_node['inputs']:
                if nodes[input_n[0]]['op']!= 'null':
                    i_node['input_layer_name'].append(nodes[input_n[0]]['name'])
            network_layers.append(i_node)
    
    
    
    node_dict=dict()
    node_dict['input_node'] = Input(shape=input_shape,name='input_node')
    for i_node in network_layers:
        
        
        if i_node['input_layer_name']:
            if len(i_node['input_layer_name'])==1:
                current_input_node=node_dict[i_node['input_layer_name'][0]]
            else:
                current_input_node=[node_dict[i] for i in i_node['input_layer_name']]
        else:
            current_input_node=node_dict['input_node']
        node_name=str(i_node['name'])
        op_type=i_node['op']
        
        layer_para=dict()
        layer_para['name']=node_name
        
        if verbose:
            if type(current_input_node).__name__ is 'Tensor':
                print(node_name+'. Input shape: '+str(current_input_node.get_shape()._dims))
            else:
                print(node_name)
            
        if 'attr' in i_node:
            layer_spec=i_node['attr']
        else:
            layer_spec=[]
            
        
        if op_type=='Convolution':
            layer_para['filters']=int(layer_spec['num_filter'])
            layer_para['strides']=np.fromstring(str(layer_spec['stride'].strip("()")),sep=', ').astype(int)
            layer_para['use_bias']=not str_to_bool(layer_spec['no_bias'])
            layer_para['kernel_size']=np.fromstring(str(layer_spec['kernel'].strip("()")),sep=', ').astype(int)
            layer_para['padding']='same'
            
            if layer_para['use_bias']:
                layer_para['weights']=[params[node_name+'_weight'].transpose((2,3,1,0)),params[node_name+'_bias']]
            else:
                layer_para['weights']=[params[node_name+'_weight'].transpose((2,3,1,0))]
            
            if 'dilate' in layer_spec:
                layer_para['dilation_rate']=np.fromstring(str(layer_spec['dilate'].strip("()")),sep=', ').astype(int)
            current_node=Conv2D(**layer_para)(current_input_node)
                
            
        
        elif op_type=='Activation':
            current_node=Activation(layer_spec['act_type'])(current_input_node)
            
            
            
        elif op_type=='BatchNorm':
            layer_para['epsilon']=np.fromstring(str(layer_spec['eps'].strip("()")),sep=', ')
            layer_para['axis']=3
            beta_init=params[node_name+'_beta']
            gamma_init=params[node_name+'_gamma']
            layer_para['weights']=[gamma_init,beta_init,np.zeros(beta_init.shape,dtype=np.float32),np.ones(beta_init.shape,dtype=np.float32)]
            current_node=BatchNormalization(**layer_para)(current_input_node)
            
            
            
        elif op_type=='elemwise_add':
            current_node=Add()(current_input_node)
            
            
            
        elif op_type=='Pooling':
            global_pool_bool=str_to_bool(layer_spec['global_pool'])
            pool_type=layer_spec['pool_type']
            if global_pool_bool:
                output_shape=(1,1,int(current_input_node.get_shape()._dims[3]))
                print(output_shape)
                if pool_type=='avg':
                    current_node=Lambda(lambda x:K.mean(x,axis=[1,2],keepdims=True), output_shape=output_shape)(current_input_node)
                elif pool_type=='max':
                    current_node=Lambda(lambda x:K.max(x,axis=[1,2],keepdims=True), output_shape=output_shape)(current_input_node)
                else:
                    raise ValueError()
                    
                current_node
            else:
                if pool_type=='avg':
                    current_node=AveragePooling2D(**layer_para)(current_input_node)
                elif pool_type=='max':
                    current_node=MaxPooling2D(**layer_para)(current_input_node)
                else:
                    raise ValueError()
                
                
                
        elif op_type=='FullyConnected':
            layer_para['units']=int(layer_spec['num_hidden'])
            layer_para['use_bias']=not str_to_bool(layer_spec['no_bias'])
            if layer_para['use_bias']:
                layer_para['weights']=[params[node_name+'_weight'].transpose(),params[node_name+'_bias']]
            else:
                layer_para['weights']=[params[node_name+'_weight'].transpose()]
                
            current_node=Dense(**layer_para)(current_input_node)
            
            
            
        elif op_type=='SoftmaxOutput':
            current_node=Activation('softmax')(current_input_node)
            
            
            
        elif op_type=='Concat':
            layer_para['axis']=3
            current_node=Concatenate(**layer_para)(current_input_node)
            
        
        
        elif op_type=='Dropout':
            layer_para['rate']=float(layer_spec['p'])
            current_node=Dropout(**layer_para)(current_input_node)
            
            
            
        elif op_type=='Flatten':
            current_node=Flatten(**layer_para)(current_input_node)
            
            
            
        else:
            raise ValueError('Unsupported layer convension: '+op_type)
            
        node_dict[node_name]=current_node
    
    keras_model=Model(inputs=node_dict['input_node'], outputs=current_node)
    return keras_model
