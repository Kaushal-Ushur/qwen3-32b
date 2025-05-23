INPUT_SCHEMA = {
    "prompt": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["What is deep learning?"]
    },
    "top_p": {
        'datatype': 'FP32',
        'required': False,
        'shape': [1],
        'example': [0.1]
    },
    "repetition_penalty": {
        'datatype': 'FP32',
        'required': False,
        'shape': [1],
        'example': [1.18]
    },
    "max_new_tokens": {
        'datatype': 'INT16',
        'required': False,
        'shape': [1],
        'example': [128]
    },
    "top_k":{
        'datatype': 'INT8',
        'required': False,
        'shape': [1],
        'example': [40]
    }
}
