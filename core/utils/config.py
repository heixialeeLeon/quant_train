from importlib import import_module
import tempfile
import os.path as osp
import shutil
import sys
from easydict import EasyDict

def get_config(config_file):
    with tempfile.TemporaryDirectory() as temp_config_dir:
        temp_config_file = tempfile.NamedTemporaryFile(
            dir=temp_config_dir, suffix='.py')
        temp_config_name = osp.basename(temp_config_file.name)
        shutil.copyfile(config_file,
                        osp.join(temp_config_dir, temp_config_name))
        temp_module_name = osp.splitext(temp_config_name)[0]
        sys.path.insert(0, temp_config_dir)
        mod = import_module(temp_module_name)
        sys.path.pop(0)
        cfg_dict = {
            name: value
            for name, value in mod.__dict__.items()
            if not name.startswith('__')
        }
        # delete imported module
        del sys.modules[temp_module_name]
        # close temp file
        temp_config_file.close()
        return EasyDict(cfg_dict)