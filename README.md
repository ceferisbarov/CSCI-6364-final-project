# Spelling correction via Deep Ensembles for English language
**Note:** This repo is a fork of the following: https://github.com/ceferisbarov/spelling-correction-tf

```sh
uv venv .env

source .env/bin/activate

uv pip install -r requirements.txt

python3 scripts/train.py

python3 scripts/evaluate_ensemble.py

python3 scripts/evaluate_symspell.py
```
Results are stored in `results` folder.
