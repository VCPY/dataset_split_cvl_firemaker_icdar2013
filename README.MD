# Dataset Split
This repository provides a split of datasets into test- and train set, where the train set is subdivided into N splits
for X-fold cross-validation.

## Datasets
This repository supports the split of three datasets: 
- CVL (https://cvl.tuwien.ac.at/research/cvl-databases/an-off-line-database-for-writer-retrieval-writer-identification-and-word-spotting/), 
- Firemaker (https://zenodo.org/record/1194612#.YkXaQDVCRhE) 
- ICDAR2013 (https://users.iit.demokritos.gr/~louloud/ICDAR2013WriterIdentificationComp/resources.html)

Place the data to split into the folder "raw_data/{dataset_name}". The dataset name must be in lowercase.
The main.py script takes two parameters. "--dataset" selects the dataset to split ("CVL", "Firemaker", "ICDAR2013"). This is needed
to determine how the label can be extracted from the file name. "-n" is the number of splits which should be applied for the train/val set. 
The resulting data is stored in the folder "result".