## DVC + Google Cloud Lab (Iris dataset)

### Setup
```bash
python -m venv venv && venv\Scripts\activate
pip install "dvc[gs]"
dvc init
mkdir data
# (download iris.csv to data/)
dvc add data/iris.csv
git add data/iris.csv.dvc .gitignore .dvc/config
git commit -m "Track iris with DVC"
dvc remote add -d myremote gs://dhanushlab
dvc remote modify myremote credentialpath "C:\Users\dhanu\OneDrive\Documents\mlopslab-476800-7644e65fe0c6.json"
git commit -am "Configure remote"
dvc push
