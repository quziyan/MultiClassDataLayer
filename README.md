# MultiClassDataLayer
These code follow the data providers in papers:
  <Learning a Metric Embedding for Face Recognition using the Multibatch Method>
  and
  <Improved Deep Metric Learning with Multi-class N-pair Loss Objective>
  
Usage:  
name: "GoogleNet"
layer {
    name: "data_mb"
    type: "MultibatchData"
    top: "data_mb"
    top: "label_type_mb"
    include {
        phase: TRAIN
    }
    transform_param {
        mirror: true
        crop_size: 224
        mean_value: 104
        mean_value: 117
        mean_value: 123
    }
    multi_batch_data_param {
        root_folder: "xxxxxxxx"
        source: "./train.txt.exist"
        batch_size: 120
        shuffle: true
        new_height: 224
        new_width: 224
        # how many identities in one batch
        identity_num_per_batch: 60
        # how many images belong to one identity
        img_num_per_identity: 2
        # Sampling strategy
        # TRUE FOR each identity has the same sampling probability
        # FALSE FOR each image has the same sampling probability
        rand_identity: true
    }
}
