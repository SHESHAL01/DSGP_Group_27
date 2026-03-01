---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:99291
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: 'Fundamentals of Generative AI for Beginners. Skills: data analysis
    machine learning data processing'
  sentences:
  - 'React, TypeScript, Redux, StyledComponents: Build Sudoku App. Skills: redux react
    typescript'
  - 'Machine Learning: Data Analysis 2017. Skills: data analysis machine learning'
  - 'Webの新しいデザイン「CSS3」入門. Skills: html css3 html5'
- source_sentence: 'Full Stack Data Science Course - Complete 2020 Edition. Skills:
    data science machine learning'
  sentences:
  - 'Modern Web Design HTML5 CSS3 beginners guide to Websites. Skills: html css'
  - 'Web Development With Google Maps. Skills: css html iis jquery javascript'
  - 'The Complete Machine Learning 2020|Python, Math|Dummy To Pro. Skills: numpy python
    data science machine learning matplotlib pandas'
- source_sentence: 'Meta Back-End Developer Professional Certificate. Skills: django
    (web framework) api endpoints mysql html production environments javascript cascading
    style sheets (css) version control github bash (unix shell) web development linux
    data structure'
  sentences:
  - 'JavaScript for Beginners Specialization. Skills: web interactivty jquery data
    manipulation javascript web interactivty jquery data manipulation javascript'
  - 'Full stack web development and AI with Python (Django). Skills: css aws html
    python data science django git javascript ai linux keras pandas'
  - 'Developing Front-End Apps with React. Skills: react (web framework) front-end
    development web development javascript user interface'
- source_sentence: 'Angular and Node.js Integration. Skills: node angular'
  sentences:
  - 'React JS - A Complete Guide for Frontend Web Development. Skills: css html react
    javascript es6'
  - 'Learn To Build Apps Using NodeJS and Angular. Skills: mongodb node angular'
  - 'Complete React Hooks Course 2020: A - Z ( Scratch to React ). Skills: jest redux
    react'
- source_sentence: 'Oracle PL/SQL is My Game: EXAM 1Z0-144. Skills: sql oracle'
  sentences:
  - 'Create a Python Application to connect to multiple databases. Skills: oracle
    sql server python sqlite mysql sql postgresql'
  - 'Make a Responsive Portfolio Website: JavaScript and HTML. Skills: javascript
    html'
  - 'The Complete MySQL Developer Course. Skills: php mysql'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Oracle PL/SQL is My Game: EXAM 1Z0-144. Skills: sql oracle',
    'Create a Python Application to connect to multiple databases. Skills: oracle sql server python sqlite mysql sql postgresql',
    'The Complete MySQL Developer Course. Skills: php mysql',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.8592, 0.4057],
#         [0.8592, 1.0000, 0.5261],
#         [0.4057, 0.5261, 1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 99,291 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                         | sentence_1                                                                         |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             |
  | details | <ul><li>min: 12 tokens</li><li>mean: 24.31 tokens</li><li>max: 82 tokens</li></ul> | <ul><li>min: 10 tokens</li><li>mean: 24.13 tokens</li><li>max: 82 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                     | sentence_1                                                                             |
  |:---------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------|
  | <code>The Complete Front-End Web Development Course!. Skills: css bootstrap html jquery javascript</code>      | <code>What is HTML? and a bit about CSS. Skills: html css</code>                       |
  | <code>Modern Web Design Beginners HTML CSS JavaScript 25+ Projects. Skills: javascript css html</code>         | <code>Create An Auction Site From Scratch: PHP and MySQLi. Skills: php html css</code> |
  | <code>Build Modern Responsive Website With HTML5, CSS3 & Bootstrap. Skills: bootstrap jquery css3 html5</code> | <code>Learn HTML5 and CSS3 from scratch. Skills: css3 html5</code>                     |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `num_train_epochs`: 4
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 4
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step  | Training Loss |
|:------:|:-----:|:-------------:|
| 0.1611 | 500   | 1.8741        |
| 0.3223 | 1000  | 1.4995        |
| 0.4834 | 1500  | 1.4534        |
| 0.6445 | 2000  | 1.451         |
| 0.8057 | 2500  | 1.4131        |
| 0.9668 | 3000  | 1.4164        |
| 1.1279 | 3500  | 1.3969        |
| 1.2891 | 4000  | 1.4002        |
| 1.4502 | 4500  | 1.3925        |
| 1.6113 | 5000  | 1.3926        |
| 1.7725 | 5500  | 1.375         |
| 1.9336 | 6000  | 1.3952        |
| 2.0947 | 6500  | 1.3869        |
| 2.2559 | 7000  | 1.368         |
| 2.4170 | 7500  | 1.3664        |
| 2.5782 | 8000  | 1.3799        |
| 2.7393 | 8500  | 1.3647        |
| 2.9004 | 9000  | 1.3667        |
| 3.0616 | 9500  | 1.3653        |
| 3.2227 | 10000 | 1.3416        |
| 3.3838 | 10500 | 1.3498        |
| 3.5450 | 11000 | 1.3523        |
| 3.7061 | 11500 | 1.3562        |
| 3.8672 | 12000 | 1.3759        |


### Framework Versions
- Python: 3.9.13
- Sentence Transformers: 5.1.2
- Transformers: 4.57.3
- PyTorch: 2.8.0+cpu
- Accelerate: 1.10.1
- Datasets: 4.4.2
- Tokenizers: 0.22.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->