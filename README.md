# Project structure

| **folder**     | **purpose**                                                                                                    |
|----------------|----------------------------------------------------------------------------------------------------------------|
| checkpoints    | for logging automatically generated checkpoints                                                                |
| config         | yaml file with project's setup parameters                                                                      |
| data           | data folder                                                                                                    |
| data/raw       | should include `george` and `non_george` folders with corresponding images and `geoges.csv`, `non_georges.csv` |
| data/processed | for test/train/validate csvs                                                                                   |
| data/predict   | for images that you want to classify                                                                           |
| data/predicted | classifier's output                                                                                            |
| models         | for logging automatically generated models after training and validation                                       |
| runs           | for logging tesorboard info                                                                                    |
| src            | code sources                                                                                                   |
| src/data       | dataset class and some utility functions for generation overall dataset and further sample                     |
| src/models     | include models' classes                                                                                        |

# How to launch the project

0. * Place the `george` and `non_george` folders with corresponding images and csvs into `/data/raw` directory. <br/>
   * Determine in `config.yaml` `mode: none`
   * Uncomment in `main` the following methods `label_and_merge` and `generate_samples`   
   * Run `main` <br/>
      * the method `label_and_merge` will create `/data/raw/data.csv` (overall dataset) and `data/processed/train.csv`
      * the method `generate_samples` will create `/data/processed/train.csv` and `/data/processed/validate.csv`

1. Determine in `config.yaml` `mode: train_and_validate`, then run `main`. <br/>
   **Note**: I achieved 0.73 accuracy on test set with the parameters from `config.yaml`
   
   As an output you will have 
      * automatically generated checkpoints in `/checpoints` directory
      * automatically generated `models/date/experiment_name/model_name/extras/model.pth`<br/>
        (e.g. `models/2023-04-04/st.george/tiny_vgg/10_epochs_batch32`)
      * automatically generated logs for tensorboard in `runs/date/experiment_name/model_name/extras/model.pth`<br/>
        (e.g. `runs/2023-04-04/st.george/tiny_vgg/10_epochs_batch32`)<br/>
        **Note**: in repository you can find an example and launch it (see below [How to lunch tensorboard](https://github.com/aimedvedeva/St.GeorgeClassifier#how-to-lunch-tensorboard)) 
   
2. * Determine in `config.yaml` `mode: load_and_eval`
   * Determine in `config.yaml` `model_to_load_path` 
   * Run `main`
   The output with the model's accuracy on train set will be in terminal

**Note**: you can repeat steps 1,2 many times to experiment with different parameters
              Keep in mind the template for the logs' paths:  `../date/experiment_name/model_name/extras/..`
              and change it properly to avoid overwriting earlier saved models, checkpoints and runs

3. * Determine in `config.yaml` `mode: make_predictions`
   * Determine in `config.yaml` `model_to_load_path`
   * Place desired images in `/data/predict` folder
   * Run `main`
   
   Output is a bunch of images with predicted labels in `data/predicted` folder

# How to lunch tensorboard 
1. open cmd
2. go to the `project_folder/venv/Scripts`, activate venv 
3. return to the `project_folder` and execute `tensorboard --logdir "runs"`
4. then go to localhost

# Result acquired on my CPU computer
With my tiny_vgg model I used SGD with lr=0.003 and batch_size=32, after 10 epochs the accuracy became higher than 0.7, the dataset is balanced. 
**Loss**
![Loss](https://github.com/aimedvedeva/St.GeorgeClassifier/blob/master/report/loss.png)
**Accuracy**
![Accuracy](https://github.com/aimedvedeva/St.GeorgeClassifier/blob/master/report/accuracy.png)

**Several classification examples**
