pip install -r requirements/initial_requirements.txt
pip install -r requirements/requirements.txt
pip install -r requirements/requirements-wandb.txt
python tools/preprocess_data.py --input /mnt/tank/c4/en/c4-train.00002-of-01024.json.gz --tokenizer-type GPT2BPETokenizer --vocab-file data/gpt2-vocab.json --merge-file data/gpt2-merges.txt --dataset-impl mmap --output-prefix data/tokenized_data --num-docs 356317
CUDA_VISIBLE_DEVICES=0 python ./deepy.py train.py -d configs 125M.yml local_setup.yml