# Introduction

This file is meant to serve both as my own notebook, and as a 'guide' for creating configs with 
AllenNLP. I document items, issues, and other important things that arise as I create the configs 
for the dataset readers.

### ReCoRD Reader

From the paper:

```
Zhang, Sheng, et al. "Record: Bridging the gap between human and machine commonsense reading comprehension." arXiv preprint arXiv:1810.12885 (2018).
```

#### Summary Of Steps:

1. Create a `.jsonnet` file 

    *Note:* If on Windows, you need to **treat** the file like it as normal `.json` file. This means no `local` variables!

2. Create the `dataset_reader` key with values
```json
{
    "type": "superglue_record",
    "transformer_model_name": "bert-base-uncased",
    "length_limit": 256
}
```  
The `type` **must** have the name that you gave your reader when decorating the class with the 
`@DatasetReader.register`. In our case here it is just `superglue_record`. The remaining entries 
correspond to arguments of `__init__`.

The config should now look like (from a top level, excluded stuff for readability)
```json
{
    "dataset_reader": {}
}
```

3. Next, outside of the Add in the `train_data_path` and `validation_data_path` to the TOP level of 
the config. The values for these keys can either be a link to a file for download, or a file on your 
local machine. 

**Note:** You need to download the files manually from [here](https://sheng-z.github.io/ReCoRD-explorer/) because Google Drive does not play nice!

For ReCoRD, this looks like
```json
"train_data_path": "data/record/train.json",
"validation_data_path": "data/record/dev.json",
```

Now our config looks like (from a top level, excluded stuff for readability):

```json
{
    "dataset_reader": {},
    "train_data_path": "",
    "validation_data_path": ""
}
```

4. You can use the rest of the [`transformer_qa`](
https://github.com/allenai/allennlp-models/blob/main/training_config/rc/transformer_qa.jsonnet)
config to complete the record config. 

5. To now run this model with `allennlp train` use the command:
```commandline
allennlp train configs/debug_record_reader.jsonnet -f -s model --include-package src.readers.superglue
```
The first argument `configs/record_reader.jsonnet` points to the config file. The second two 
arguments `-f -s model` are for saving the model to a directory. The `-s model` indicates the 
serialization dir, while `-f` says to force overwriting that directory. The final argument 
`--include-package src.readers.superglue` includes the package.

#### Full steps taken (Also My notes): 

Since the `record_reader.py` is based on `transformer_squad.py` I started with the same `.jsonnet` 
config as they did. That can be found 
[here](https://github.com/allenai/allennlp-models/blob/main/training_config/rc/transformer_qa.jsonnet).

The tutorial does not specify this, but I think you need to import the package that has the reader 
in it.

You do need to include the package with the `--include-package` cli argument