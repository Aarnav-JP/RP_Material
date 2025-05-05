---
base_model: sentence-transformers/all-MiniLM-L6-v2
library_name: sentence-transformers
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:639
- loss:TripletLoss
widget:
- source_sentence: schnitzler's film has a great hook , some clever bits and well-drawn
    , if standard issue , characters , but is still only partly satisfying .
  sentences:
  - the vampire thriller blade ii starts off as a wild hoot and then sucks the blood
    out of its fun – toward the end , you can feel your veins cringing from the workout
    .
  - there are some movies that hit you from the first scene and you know it's going
    to be a trip . igby goes down is one of those movies .
  - conceptually brilliant . . . plays like a living-room war of the worlds , gaining
    most of its unsettling force from the suggested and the unknown .
- source_sentence: this film was made to get laughs from the slowest person in the
    audience -- just pure slapstick with lots of inane , inoffensive screaming and
    exaggerated facial expressions .
  sentences:
  - charles' entertaining film chronicles seinfeld's return to stand-up comedy after
    the wrap of his legendary sitcom , alongside wannabe comic adams' attempts to
    get his shot at the big time .
  - '" men in black ii , " has all the earmarks of a sequel . the story is less vibrant
    , the jokes are a little lukewarm , but will anyone really care ?'
  - these are textbook lives of quiet desperation .
- source_sentence: the movie is too amateurishly square to make the most of its own
    ironic implications .
  sentences:
  - a beautifully observed character piece .
  - this stuck pig of a movie flails limply between bizarre comedy and pallid horror
    .
  - just about all of the film is confusing on one level or another , making ararat
    far more demanding than it needs to be .
- source_sentence: a manipulative feminist empowerment tale thinly posing as a serious
    drama about spousal abuse .
  sentences:
  - could this be the first major studio production shot on video tape instead of
    film ?
  - this formulaic chiller will do little to boost stallone's career .
  - scherfig , the writer-director , has made a film so unabashedly hopeful that it
    actually makes the heart soar . yes , soar .
- source_sentence: parker probably thinks he's shaking up a classic the way kenneth
    branagh and baz luhrmann have , but this half-hearted messing-about just makes
    us miss wilde's still-contemporary play .
  sentences:
  - this ecologically minded , wildlife friendly film teaches good ethics while entertaining
    with its unconventionally wacky but loving family
  - like all abstract art , the film does not make this statement in an easily accessible
    way , and -- unless prewarned -- it would be very possible for a reasonably intelligent
    person to sit through its tidal wave of imagery and not get this vision at all
    .
  - my precious new star wars movie is a lumbering , wheezy drag . . .
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
    "parker probably thinks he's shaking up a classic the way kenneth branagh and baz luhrmann have , but this half-hearted messing-about just makes us miss wilde's still-contemporary play .",
    'like all abstract art , the film does not make this statement in an easily accessible way , and -- unless prewarned -- it would be very possible for a reasonably intelligent person to sit through its tidal wave of imagery and not get this vision at all .',
    'this ecologically minded , wildlife friendly film teaches good ethics while entertaining with its unconventionally wacky but loving family',
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

* Size: 639 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>sentence_2</code>
* Approximate statistics based on the first 639 samples:
  |         | sentence_0                                                                        | sentence_1                                                                        | sentence_2                                                                        |
  |:--------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                            | string                                                                            |
  | details | <ul><li>min: 5 tokens</li><li>mean: 28.27 tokens</li><li>max: 64 tokens</li></ul> | <ul><li>min: 5 tokens</li><li>mean: 28.13 tokens</li><li>max: 58 tokens</li></ul> | <ul><li>min: 6 tokens</li><li>mean: 28.69 tokens</li><li>max: 64 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                     | sentence_1                                                                                                                                                                                                   | sentence_2                                                                                                                                                                                        |
  |:-----------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>idiotic and ugly .</code>                                                                | <code>you will emerge with a clearer view of how the gears of justice grind on and the death report comes to share airtime alongside the farm report .</code>                                                | <code>just like the deli sandwich : lots of ham , lots of cheese , with a sickly sweet coating to disguise its excrescence until just after ( or during ) consumption of its second half .</code> |
  | <code>real women may have many agendas , but it also will win you over , in a big way .</code> | <code>this film seems thirsty for reflection , itself taking on adolescent qualities .</code>                                                                                                                | <code>director yu seems far more interested in gross-out humor than in showing us well-thought stunts or a car chase that we haven't seen 10 , 000 times .</code>                                 |
  | <code>a benign but forgettable sci-fi diversion .</code>                                       | <code>it's rare that a movie can be as intelligent as this one is in every regard except its storyline ; everything that's good is ultimately scuttled by a plot that's just too boring and obvious .</code> | <code>argento , at only 26 , brings a youthful , out-to-change-the-world aggressiveness to the project , as if she's cut open a vein and bled the raw film stock .</code>                         |
* Loss: <code>__main__.TripletLoss</code> with these parameters:
  ```json
  {
      "distance_metric": "TripletDistanceMetric.EUCLIDEAN",
      "triplet_margin": 5
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