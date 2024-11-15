#! /bin/bash

cd $HOME

sudo apt-get update
sudo apt-get install -y zsh
# See https://github.com/ohmyzsh/ohmyzsh#unattended-install
yes | sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

pushd $HOME/dotfiles && ./setup.sh && popd

if [ ! -d "" ]; then # blank for proprietary reasons
    git clone -q git@github.com:!!!!/!!!!.git # blank for proprietary reasons
    pushd repo && git config receive.denyCurrentBranch ignore && popd
fi

cp $HOME/.env $HOME/repo/.env

# pyenv
curl https://pyenv.run | bash
sudo apt-get install -y libssl-dev liblzma-dev libsqlite3-dev make build-essential zlib1g-dev libbz2-dev libreadline-dev libffi-dev wget curl llvm
~/.pyenv/bin/pyenv install 3.11

# Add ssh 
eval `ssh-agent -s`
ssh-add ~/.ssh/id_ed25519
ssh-add -L

# Poetry
curl -sSL https://install.python-poetry.org | python3 -
~/.local/bin/poetry config virtualenvs.in-project true
pushd $HOME/repo && ~/.local/bin/poetry env use ~/.pyenv/versions/3.11.10/bin/python && ~/.local/bin/poetry install --without dev && popd
