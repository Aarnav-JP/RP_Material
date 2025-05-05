---
base_model: sentence-transformers/all-MiniLM-L6-v2
library_name: sentence-transformers
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:638
- loss:CachedMultipleNegativesRankingLoss
widget:
- source_sentence: 'you see robert de niro singing - and dancing to - west side story
    show tunes . choose your reaction : a . ) that sure is funny ! b . ) that sure
    is pathetic !'
  sentences:
  - automatically pegs itself for the straight-to-video sci-fi rental shelf .
  - serving sara should be served an eviction notice at every theater stuck with it
    .
  - a movie that's about as overbearing and over-the-top as the family it depicts
    .
- source_sentence: the cumulative effect of watching this 65-minute trifle is rather
    like being trapped while some weird relative trots out the video he took of the
    family vacation to stonehenge . before long , you're desperate for the evening
    to end .
  sentences:
  - my precious new star wars movie is a lumbering , wheezy drag . . .
  - farrell . . . thankfully manages to outshine the role and successfully plays the
    foil to willis's world-weary colonel .
  - jonathan parker's bartleby should have been the be-all-end-all of the modern-office
    anomie films .
- source_sentence: fred schepisi's tale of four englishmen facing the prospect of
    their own mortality views youthful affluence not as a lost ideal but a starting
    point .
  sentences:
  - everyone's insecure in lovely and amazing , a poignant and wryly amusing film
    about mothers , daughters and their relationships .
  - director rob marshall went out gunning to make a great one .
  - there's something deeply creepy about never again , a new arrow in schaeffer's
    quiver of ineptitudes .
- source_sentence: for its 100 minutes running time , you'll wait in vain for a movie
    to happen .
  sentences:
  - parker probably thinks he's shaking up a classic the way kenneth branagh and baz
    luhrmann have , but this half-hearted messing-about just makes us miss wilde's
    still-contemporary play .
  - avary's film never quite emerges from the shadow of ellis' book .
  - it may seem long at 110 minutes if you're not a fan , because it includes segments
    of 12 songs at a reunion concert .
- source_sentence: the film has too many spots where it's on slippery footing , but
    is acceptable entertainment for the entire family and one that's especially fit
    for the kiddies .
  sentences:
  - the director has injected self-consciousness into the proceedings at every turn
    . the results are far more alienating than involving .
  - the laser-projected paintings provide a spell-casting beauty , while russell and
    dreyfus are a romantic pairing of hearts , preciously exposed as history corners
    them .
  - will certainly appeal to asian cult cinema fans and asiaphiles interested to see
    what all the fuss is about .
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision fa97f6e7cb1a59073dff9e6b13e2715cf7475ac9 -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
ModifiedSentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel 
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
    "the film has too many spots where it's on slippery footing , but is acceptable entertainment for the entire family and one that's especially fit for the kiddies .",
    'the director has injected self-consciousness into the proceedings at every turn . the results are far more alienating than involving .',
    'will certainly appeal to asian cult cinema fans and asiaphiles interested to see what all the fuss is about .',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
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

* Size: 638 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 638 samples:
  |         | sentence_0                                                                        | sentence_1                                                                        | label                                           |
  |:--------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:------------------------------------------------|
  | type    | string                                                                            | string                                                                            | int                                             |
  | details | <ul><li>min: 5 tokens</li><li>mean: 28.29 tokens</li><li>max: 64 tokens</li></ul> | <ul><li>min: 5 tokens</li><li>mean: 28.25 tokens</li><li>max: 64 tokens</li></ul> | <ul><li>0: ~50.00%</li><li>1: ~50.00%</li></ul> |
* Samples:
  | sentence_0                                                                                                                       | sentence_1                                                                                                                                                             | label          |
  |:---------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------|
  | <code>an imponderably stilted and self-consciously arty movie .</code>                                                           | <code>feels like the grittiest movie that was ever made for the lifetime cable television network .</code>                                                             | <code>0</code> |
  | <code>an undeniably gorgeous , terminally smitten document of a troubadour , his acolytes , and the triumph of his band .</code> | <code>these are textbook lives of quiet desperation .</code>                                                                                                           | <code>1</code> |
  | <code>it leers , offering next to little insight into its intriguing subject .</code>                                            | <code>stands as one of the year's most intriguing movie experiences , letting its imagery speak for it while it forces you to ponder anew what a movie can be .</code> | <code>0</code> |
* Loss: <code>__main__.CachedMultipleNegativesRankingLoss</code> with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 128
- `per_device_eval_batch_size`: 128
- `num_train_epochs`: 1
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 128
- `per_device_eval_batch_size`: 128
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
- `num_train_epochs`: 1
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
- `use_ipex`: False
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
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
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
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch | Step |
|:-----:|:----:|
| 1.0   | 5    |


### Framework Versions
- Python: 3.12.2
- Sentence Transformers: 3.4.0
- Transformers: 4.48.1
- PyTorch: 2.4.1
- Accelerate: 1.3.0
- Datasets: 3.2.0
- Tokenizers: 0.21.0

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