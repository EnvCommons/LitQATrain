# Data Upload Requirements for LitQATrain

## Overview
This environment requires the following data to be uploaded to OpenReward cloud storage.

## Directory Structure
```
/orwd_data/
└── train.parquet
```

## File Descriptions
- **train.parquet**: Dataset containing scientific literature QA pairs with columns: `id`, `question`, `answer`, `source_doi`, `key_passage`, `domain`

## Upload Instructions
Upload `train.parquet` to the `EnvCommons/litqatrain` namespace on OpenReward.
