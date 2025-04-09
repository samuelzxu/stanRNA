git config --global user.email "samuelzxu@gmail.com"
git config --global user.name "Samuel Xu"
apt update; apt upgrade -y; apt install -y tmux nvtop htop
rm -rf rna_env
pip install uv
uv venv rna_env --python 3.11
source rna_env/bin/activate
uv pip install -r requirements.txt
huggingface-cli login
wandb login