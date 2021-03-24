# BERT_NER

## CoNLL-2003

```
poetry run python run_ner.py \
  --model_name_or_path bert-base-cased \
  --dataset_name conll2003 \
  --cache_dir models \
  --output_dir output \
  --overwrite_output_dir \
  --seed 42 \
  --do_train \
  --do_eval \
  --do_predict
```

### Eval Result

```
"eval_accuracy": 0.9913944161052919,
"eval_f1": 0.951645399597045,
```

### Test Result

```
"eval_accuracy": 0.9831377193926994,
"eval_f1": 0.9155500705218618,
```