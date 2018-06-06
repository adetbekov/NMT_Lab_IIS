``` shell
python -m nmt.nmt \
    --attention=scaled_luong \
    --src=en --tgt=kz \
    --vocab_prefix=/Users/dosya/Projects/lab_iis/nmt/nmt_data/vocab \
    --train_prefix=/Users/dosya/Projects/lab_iis/nmt/nmt_data/train \
    --out_dir=/Users/dosya/Projects/lab_iis/nmt/nmt_attention_model \
    --dev_prefix=/Users/dosya/Projects/lab_iis/nmt/nmt_data/tst_1 \
    --test_prefix=/Users/dosya/Projects/lab_iis/nmt/nmt_data/tst_2 \
    --num_train_steps=4000 \
    --steps_per_stats=100 \
    --num_layers=2 \
    --num_units=128 \
    --dropout=0.2 \
    --metrics=bleu
```

``` shell
    python -m nmt.nmt \
    --out_dir=/Users/dosya/Projects/lab_iis/nmt/nmt_attention_model \
    --inference_input_file=/Users/dosya/Projects/lab_iis/nmt/nmt_data/test.en \
    --inference_output_file=/Users/dosya/Projects/lab_iis/nmt/nmt_attention_model/output_infer
```
    