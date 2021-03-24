# BERT_NER

## CoNLL-2003

```
poetry run python run_ner.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name conll2003 \
  --cache_dir models \
  --output_dir output \
  --overwrite_output_dir \
  --seed 42 \
  --do_train \
  --do_eval \
  --do_predict
```