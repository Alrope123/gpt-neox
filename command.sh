scp xinxi@34.67.201.205:/home/xinxi/gpt-neox/data2/tokenized_data_text_validation.idx /home/xinxi/gpt-neox/data
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
bash requirements/initial_requirements.bash
pip install -r requirements/requirements.txt
pip install -r requirements/requirements-wandb.txt
wandb login c962dc2aa3adc82f9dbcc17d82f367a93fc17120
python ./megatron/fused_kernels/setup.py install
# python tools/preprocess_data.py --input /mnt/tank/c4/en/c4-train.00002-of-01024.json.gz --tokenizer-type GPT2BPETokenizer --vocab-file data/gpt2-vocab.json --merge-file data/gpt2-merges.txt --dataset-impl mmap --output-prefix data/tokenized_data --num-docs 356317
# python tools/preprocess_data.py --input c4-train --tokenizer-type GPT2BPETokenizer --vocab-file data/gpt2-vocab.json --merge-file data/gpt2-merges.txt --dataset-impl mmap --output-prefix data/tokenized_data --num-docs 365000000 --workers 176
# taskset -c 0-3 python tools/preprocess_data.py --input c4-train --input-range 0-93 --tokenizer-type GPT2BPETokenizer  --output-prefix data/tokenized_data --num-docs 32424847 --workers 4 &> logs/0-93.txt
# taskset -c 4-7 python tools/preprocess_data.py --input c4-train --input-range 93-186 --tokenizer-type GPT2BPETokenizer  --output-prefix data/tokenized_data --num-docs 32424847 --workers 4 &> logs/93-186.txt
# taskset -c 8-11 python tools/preprocess_data.py --input c4-train --input-range 186-279 --tokenizer-type GPT2BPETokenizer  --output-prefix data/tokenized_data --num-docs 32424847 --workers 4 &> logs/186-279.txt
# taskset -c 12-15 python tools/preprocess_data.py --input c4-train --input-range 279-372 --tokenizer-type GPT2BPETokenizer  --output-prefix data/tokenized_data --num-docs 32424847 --workers 4 &> logs/279-372.txt
# taskset -c 16-19 python tools/preprocess_data.py --input c4-train --input-range 372-465 --tokenizer-type GPT2BPETokenizer  --output-prefix data/tokenized_data --num-docs 32424847 --workers 4 &> logs/372-465.txt
# taskset -c 20-23 python tools/preprocess_data.py --input c4-train --input-range 465-558 --tokenizer-type GPT2BPETokenizer  --output-prefix data/tokenized_data --num-docs 32424847 --workers 4 &> logs/465-558.txt
# taskset -c 24-27 python tools/preprocess_data.py --input c4-train --input-range 558-651 --tokenizer-type GPT2BPETokenizer  --output-prefix data/tokenized_data --num-docs 32424847 --workers 4 &> logs/558-651.txt
# taskset -c 28-31 python tools/preprocess_data.py --input c4-train --input-range 651-744 --tokenizer-type GPT2BPETokenizer  --output-prefix data/tokenized_data --num-docs 32424847 --workers 4 &> logs/651-744.txt
# taskset -c 32-35 python tools/preprocess_data.py --input c4-train --input-range 744-837 --tokenizer-type GPT2BPETokenizer  --output-prefix data/tokenized_data --num-docs 32424847 --workers 4 &> logs/744-837.txt
# taskset -c 36-39 python tools/preprocess_data.py --input c4-train --input-range 837-930 --tokenizer-type GPT2BPETokenizer  --output-prefix data/tokenized_data --num-docs 32424847 --workers 4 &> logs/837-930.txt
# taskset -c 40-43 python tools/preprocess_data.py --input c4-train --input-range 930-1024 --tokenizer-type GPT2BPETokenizer  --output-prefix data/tokenized_data --num-docs 32424847 --workers 4 &> logs/930-1024.txt
CUDA_VISIBLE_DEVICES=0 python ./deepy.py train.py -d configs 125M.yml local_setup.yml
python ./tools/convert_sequential_to_hf.py  --input_dir checkpoints/125M/global_step70000 --config_file checkpoints/125M/global_step70000/configs/125M.yml --output_dir hf_model/125M
