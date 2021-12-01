data/Intro to remote data storage

Tutorial is here: https://dvc.org/doc/start/data-and-model-versioning
data

0. Make sure you have the config.local file in your .dvc folder. (email) 
00. Make sure you have dvc installed locally https://dvc.org/doc/install
In command line (terminal) do the following: 
Open a terminal window
1. cd youprojectdirecty/.git/

PULL ( aka starting for the day):
1. dvc pull

ADD:
1. dvc add data/example_file_name
2. (this will be generated from step 1) git add data/example)file_name.dvc data/.gitignore
3. git commit -m "Add raw data"

PUSH: If finished with data for the day, do the following:
1. dvc push
