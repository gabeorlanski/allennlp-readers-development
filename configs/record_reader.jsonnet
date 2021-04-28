{
  "dataset_reader": {
    "type": "superglue_record",
    "transformer_model_name": "bert-base-uncased",
    "length_limit": 256
  },
  "train_data_path": "data/record/train.json",
  "validation_data_path": "data/record/dev.json",
  "vocabulary": {
    "type": "empty"
  },
  "model": {
    "type": "transformer_qa",
    "transformer_model_name": "bert-base-uncased"
  },
  "data_loader": {
    "type": "multiprocess",
    "num_workers": 4,
    "batch_sampler": {
      "type": "bucket",
      "batch_size": 4
    }
  },
  "trainer": {
    "optimizer": {
      "type": "huggingface_adamw",
      "weight_decay": 0.0,
      "parameter_groups": [
        [["bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0}]
      ],
      "lr": 2e-5,
      "eps": 1e-8
    },
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "patience": 5
    },
    "grad_clipping": 1.0,
    "num_epochs": 10,
    "validation_metric": "+per_instance_f1"
  },
  "random_seed": 100,
  "numpy_seed": 100,
  "pytorch_seed": 100
}