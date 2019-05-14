import pickle
import numpy as np
import os
import json
from io import StringIO
from six import BytesIO


def model_fn(model_dir):
    print("model_dir:{}".format(model_dir))
    with open(os.path.join(model_dir, 'model.pkl'), 'rb') as inp:
        clf = pickle.load(inp)
        return clf


def input_fn(request_body, request_content_type):
    
    print("request_content_type :{}".format(request_content_type))
    
    if request_content_type == 'text/csv':
        # Read the raw input data as CSV.
        npreq = np.genfromtxt(StringIO(request_body), delimiter=",")
        print ("shape of array:{}".format(npreq.shape))
        if npreq.ndim == 1:
            npreq = np.array([npreq])
        return npreq
    elif request_content_type == 'application/x-npy':
        return _npy_loads(request_body)
    else:
        raise ValueError("{} not supported by script!".format(request_content_type))


def output_fn(prediction, response_content_type):
    
    print("response_content_type :{}".format(response_content_type))
    
    if response_content_type == "application/x-npy":
        return _npy_dumps(prediction), 'application/x-npy'
    elif response_content_type == 'text/csv':
        output = StringIO()    
        np.savetxt(output, prediction, delimiter='\n')
        return output.getvalue()
    else:
        raise ValueError("{} not supported by script!".format(response_content_type))    
                
def predict_fn(input_data, model):
    prediction = model.predict(input_data)
    return np.array([prediction])

def _npy_loads(data):
    """
    Deserializes npy-formatted bytes into a numpy array
    """
    stream = BytesIO(data)
    return np.load(stream)

def _npy_dumps(data):
    """
    Serialized a numpy array into a stream of npy-formatted bytes.
    """
    buffer = BytesIO()
    np.save(buffer, data)
    return buffer.getvalue()
    