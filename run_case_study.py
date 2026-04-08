import importlib
import sys
from logging import getLogger
from recbole.config import Config
from recbole.data import (
    create_dataset,
    data_preparation
)
from recbole.data.transform import construct_transform
from recbole.utils import (
    init_logger,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
    case_study
)

def get_model(model_name):
    module_path = '.'.join(['recbole_model', model_name])
    model_module = importlib.import_module(module_path, __name__)
    model_class = getattr(model_module, model_name)
    return model_class

def run_recbole(model, dataset, config_file_list, saved=True):
    model = get_model(model)
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list
    )
    init_seed(config["seed"], config["reproducibility"])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = model(config, train_data._dataset).to(config["device"])
    logger.info(model)

    transform = construct_transform(config)
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # trainer loading and initialization
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    # trainer = CustomTrainer(config, model)  # 使用自定义的Trainer

    # model training
    # best_valid_score, best_valid_result = trainer.fit(
    #     train_data, valid_data, saved=saved, show_progress=config["show_progress"]
    # )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config["show_progress"])
    
    import numpy as np
    import torch

    case_full_scores = case_study.full_sort_scores(test_data.dataset['user_id'], model, test_data, device=config["device"])
    case_topk_ = case_study.full_sort_topk(
        test_data.dataset['user_id'], model, test_data, k=1, device=config["device"]
    )

    #logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")
    print_result(test_result, logger, k=4)

    return {
        #"best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        #"best_valid_result": best_valid_result,
        "test_result": test_result,
    }

def print_result(test_result, logger, k=4):
    count = 0
    info = '\ntest result:'
    for i in test_result.keys():
        if count == 0:
            info += '\n'
        count = (count + 1) % k
        info += "{:15}:{:<10}    ".format(i, test_result[i])
    logger.info(info)


if __name__ == '__main__':
    dataset = 'lastfm'
    model='MDPRec'
    run_recbole(model=model, dataset=dataset,
                config_file_list=['./config/data.yaml', f'./config/{dataset}.yaml'])
    
# 更改模型加载路径trainer self.saved_model_file