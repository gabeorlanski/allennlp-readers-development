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
    "batch_sampler": {
      "type": "bucket",
      "batch_size": 4,
      "sorting_keys": ["question_with_context"]
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
    "num_epochs": 20,
    "grad_norm": 5.0,
    "patience": 10,
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "factor": 0.5,
      "mode": "max",
      "patience": 2
    },
    "grad_clipping": 1.0,
    "validation_metric": "+per_instance_f1"
  },
  "random_seed": 100,
  "numpy_seed": 100,
  "pytorch_seed": 100
}