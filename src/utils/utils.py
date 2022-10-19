import os
import re
import yaml
import importlib
import logging
import requests
import json
from tqdm import tqdm
from collections import OrderedDict
from .compress_file import extract_compressed_file

LOG_DIR = './log/'
SAVE_DIR = './saved/'

def set_color(log, color, highlight=True, keep=False):
    r"""Set color for log string.

    Args:
        log(str): the
    """
    if keep:
        return log
    color_set = ['black', 'red', 'green',
                 'yellow', 'blue', 'pink', 'cyan', 'white']
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = '\033['
    if highlight:
        prev_log += '1;3'
    else:
        prev_log += '0;3'
    prev_log += str(index) + 'm'
    return prev_log + log + '\033[0m'

def color_dict(dict_, keep=True):
    key_color = 'blue'
    val_color = 'yellow'

    def color_kv(k, v, k_f, v_f):
        info = (set_color(k_f, key_color, keep=keep) + '=' +
                set_color(v_f, val_color, keep=keep)) % (k, v)
        return info

    des = 4
    if 'epoch' in dict_:
        start = set_color('Training: ', 'green', keep=keep)
        start += color_kv('Epoch', dict_['epoch'], '%s', '%3d')
    else:
        start = set_color('Testing: ', 'green', keep=keep)
    info = ' '.join([color_kv(k, v, '%s', '%.'+str(des)+'f')
                    for k, v in dict_.items() if k != 'epoch'])
    return start + ' [' + info + ']'

def color_dict_normal(dict_, keep=True):
    dict_ = OrderedDict(sorted(dict_.items()))
    key_color = 'blue'
    val_color = 'yellow'

    def color_kv(k, v, k_f, v_f):
        info = (set_color(k_f, key_color, keep=keep) + '=' +
                set_color(v_f, val_color, keep=keep)) % (k, v)
        return info
    info = '\n'.join([color_kv(k, str(v), '%s', '%s')
                     for k, v in dict_.items()])
    return info

def parser_yaml(config_path):
    loader = yaml.FullLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(
            u'''^(?:
            [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X
        ), list(u'-+0123456789.')
    )
    with open(config_path, 'r', encoding='utf-8') as f:
        ret = yaml.load(f.read(), Loader=loader)
    return ret

def get_model(model_name: str):
    model_submodule = ['language_modeling', 'knowledge_graph', 'extreme_classification']
    model_file_name = model_name.lower()
    model_module = None
    for submodule in model_submodule:
        module_path = '.'.join(['src', submodule, 'model', model_file_name])
        if importlib.util.find_spec(module_path, __name__):
            model_module = importlib.import_module(module_path, __name__)
            break

    if model_module is None:
        raise ValueError(
            f'`model_name` [{model_name}] is not the name of an existing model.')
    model_class = getattr(model_module, model_name)
    dir = os.path.dirname(model_module.__file__)
    conf = dict()
    fname = os.path.join(os.path.dirname(dir), '..', 'basemodel.yaml')
    conf.update(parser_yaml(fname))
    for name in ['all', model_file_name]:
        fname = os.path.join(dir, 'config', name+'.yaml')
        if os.path.isfile(fname):
            conf.update(parser_yaml(fname))
    return model_class, conf

def get_dataset_config(name: str):
    conf_file = os.path.join(os.path.dirname(__file__), '..', 'config', f"{name}.yaml")
    conf = parser_yaml(conf_file)
    return conf

class RemoveColorFilter(logging.Filter):
    def filter(self, record):
        if record:
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            record.msg = ansi_escape.sub('', str(record.msg))
        return True

def add_file_handler(logger: logging.Logger, log_dir:str, file_path: str, formatter: logging.Formatter = None):
    # log_file_dir = os.path.join(LOG_DIR, 'logs/')
    log_file_dir = os.path.join(log_dir, file_path)
    if not os.path.exists(log_file_dir):
        os.makedirs(log_file_dir)
    file_handler = logging.FileHandler(os.path.join(log_file_dir, file_path))
    file_handler.setLevel(logging.INFO)
    if formatter is None:
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(message)s', "%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    file_handler.addFilter(RemoveColorFilter())
    logger.addHandler(file_handler)
    return logger

def get_logger(log_dir, file_path: str = None):
    FORMAT = '[%(asctime)s] %(levelname)s %(message)s'
    logger = logging.getLogger('MIDX')

    formatter = logging.Formatter(FORMAT, "%Y-%m-%d %H:%M:%S")
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if file_path is not None:
        logger = add_file_handler(logger, log_dir, file_path, formatter)
    return logger


def get_download_url_from_recstore(share_number: str):
    headers = {
        "Host": "recapi.ustc.edu.cn",
        "Content-Type": "application/json",
    }
    data_resource_list = {
        "share_number": share_number,
        "share_resource_number": None,
        "is_rec": "false",
        "share_constraint": {}
    }
    resource = requests.post(
        'https://recapi.ustc.edu.cn/api/v2/share/target/resource/list',
        json=data_resource_list, headers=headers)
    resource = resource.text.encode("utf-8").decode("utf-8-sig")
    resource = json.loads(resource)
    resource = resource['entity'][0]['number']
    data = {
        "share_number": share_number,
        "share_constraint": {},
        "share_resources_list": [
            resource
        ]
    }
    res = requests.post(
        "https://recapi.ustc.edu.cn/api/v2/share/download",
        json=data, headers=headers)
    res = res.text.encode("utf-8").decode("utf-8-sig")
    res = json.loads(res)
    download_url = res['entity'][resource] + "&download=download"
    return download_url

def download_dataset(url: str, name: str):
    save_dir = f"./data/{name}"
    if url.startswith('http'):  # remote
        try:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            if "rec.ustc.edu.cn" in url:
                url = get_download_url_from_recstore(share_number=url.split('/')[-1])
                zipped_file_name = f"{name}.zip"
            else:
                zipped_file_name = url.split('/')[-1]
            dataset_file_path = os.path.join(save_dir, zipped_file_name)
            response = requests.get(url, stream=True)
            content_length = int(response.headers.get('content-length', 0))
            with open(dataset_file_path, 'wb') as file, \
                tqdm(desc='Downloading dataset',
                     total=content_length, unit='iB',
                     unit_scale=True, unit_divisor=1024) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)
            extract_compressed_file(dataset_file_path, save_dir)
            return save_dir
        except:
            print("Something went wrong in downloading dataset file.")
