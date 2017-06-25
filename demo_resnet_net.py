import urllib.request 
from MXNET2Keras import MXNET2Keras
import os.path

#Take official pretrained squeeze net as an example
if not ('resnet-152-0000.params'):
    urllib.request.urlretrieve('http://data.dmlc.ml/models/imagenet/resnet/152-layers/resnet-152-0000.params','resnet-152-0000.params')
    urllib.request.urlretrieve('http://data.dmlc.ml/models/imagenet/resnet/152-layers/resnet-152-symbol.json','resnet-152-symbol.json')

"""
Parse Keras model from both params and json files.
"""
resnet152_keras=MXNET2Keras(prefix='resnet-152',input_shape=(224,224,3))
