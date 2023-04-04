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

0. Place the _george_ and _non_george_ folders with corresponding images into `/data/raw` directory.
   In `main` call 
      * the method `label_and_merge` to create `/data/raw/data.csv` (overall dataset) and `data/processed/train.csv`
      * the method `generate_samples` to create `/data/processed/train.csv` and `/data/processed/validate.csv`

1. Determine in `config.yaml` mode = `train_and_validate`, then run `main`.
   **Note**: I achieved 0.73 accuracy on test set with the parameters from `config.yaml`
   
   As an output you will have 
      * automatically generated checkpoints in `/checpoints` directory
      * automatically generated `models/date/experiment_name/model_name/extras/model.pth`
        (e.g. `models/2023-04-04/st.george/tiny_vgg/10_epochs_batch32`)
      * automatically generated logs for tensorboard in `runs/date/experiment_name/model_name/extras/model.pth`
        (e.g. `runs/2023-04-04/st.george/tiny_vgg/10_epochs_batch32`)
        **Note**: in repository you can find an example and launch it (see below **How to lunch tensorboard**) 
   
2. Determine in `config.yaml` mode = `load_and_eval`, then run `main`.
   The output with the model's accuracy on train set will be in terminal

**Note**: you can repeat steps 1,2 many times to experiment with different parameters
          Keep in mind the template for the logs' paths:  `../date/experiment_name/model_name/extras/..`
          and change it properly to avoid overwriting earlier saved models, checkpoints and runs

3. * Determine in `config.yaml` mode = `make_predictions`, then run `main`
   * Place desired images in `/data/predict` folder
   
   Output is a bunch of images with predicted labels in `data/predicted` folder

# How to lunch tensorboard 
1. open cmd
2. go to the `project_folder/venv/Scripts`, activate venv 
3. return to the `project_folder` and execute `tensorboard --logdir "runs"`
4. then go to localhost