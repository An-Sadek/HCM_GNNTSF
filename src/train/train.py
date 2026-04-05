import yaml

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

class Training():
    def __init__(self, model: str, model_name: str):
        config_dict = config[train][model][model_name]
        self.batch_size = config_dict["batch_size"]
        self.epoch = config_dict["epoch"]
        self.dropout = config_dict["dropout"]
        self.learning_rate = config_dict["learning_rate"]
        self.optimizer = config_dict["optimizer"]
        self.regularization = config_dict["regularization"]

        

    
        