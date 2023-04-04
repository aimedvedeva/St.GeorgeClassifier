# Project structure

| **folder**     | **purpose**                                                                                |
|----------------|--------------------------------------------------------------------------------------------|
| checkpoints    | for logging automatically generated checkpoints                                            |
| config         | yaml file with project's setup parameters                                                  |
| data           | data folder                                                                                |
| data/raw       | should include _george_ and _non_george_ <br/> folders with corresponding images           |
| data/processed | for test/train/validate csvs                                                               |
| data/predict   | for images that you want to classify                                                       |
| data/predicted | classifier's output                                                                        |
| models         | for logging automatically generated models after training and validation                   |
| runs           | for logging tesorboard info                                                                |
| src            | code sources                                                                               |
| src/data       | dataset class and some utility functions for generation overall dataset and further sample |
| src/models     | include models' classes                                                                    |

# How to launch the project

Should set preferable parameters in `config.yaml` file. 

1. Choose the mode
  *  `train_and_validate` mode's output locates in _run_, _models_, _checkpoints_ folders
  *  `load_and_eval` mode's output is teh statement in console with eventual metrics calculated on test.csv file
      Note: you should determine `model_to_load_path` in config.yaml to clarify which model you want to evaluate
  *  `make_predictions` mode's output is a bunch of images with predicted labels in _predicted_ folder
      Note: before running the code you should place desired images in _predict_ folder
2. Choose other parameters. See the description in `config.yaml`

# How to launch tensorboard
 
1. open cmd
2. go to the `project folder/venv/Scripts`, activate venv 
3. return to the `project folder` and execute `tensorboard --logdir "runs"`
4. then go to localhost
