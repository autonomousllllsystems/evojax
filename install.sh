

git checkout branch
pip install -e .

cd examples

python train_slimevolley.py --gpu-id 0 --resume ./log/slimevolley/best.npz
python train_slimevolley.py --gpu-id 0 --max-iter 1 --log_dir ./log/test --debug
python train_slimevolley.py --gpu-id 0 --max-iter 1 --log_dir ./log/stdsolver --debug


python train_slimevolley.py --gpu-id 0 --max-iter 50 --log_dir ./log/test