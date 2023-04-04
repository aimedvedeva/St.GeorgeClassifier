import os

import yaml

CONFIG_PATH = "config\\"

def load_config(project_path, config_name):
    with open(os.path.join(project_path, CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config
