### EEG Preprocessing Pipeline Tutorial 

## First part in MATLAB 

## Second part on your local machine 

# Step 1: Clone / download the repository from GitHub

# Structure of the repository 

PreprocessingPipeline/
├── preprocessing/
│   ├── __init__.py
│   ├── rename_channels.py
│   ├── mutual_alignment.py
│   └── individual_preprocessing.py
├── requirements.txt # you need this 
├── utils/
│   ├── __init__.py
└── run_pipeline.py # and this 

# Step 2: In your terminal go to the repository 

![Example of my (M. Glushanina) PC, you should see smth like this](image.png)

# Step 3: Install all the necessary requirements 

To install all the requirements using pip: 

```
pip install -r requirements.txt 
```

# Step 4: Run the code on your local machine 

The structure of the code is 

```
python run_pipeline.py file1 file2 name1 name2 --montage montage_file 
```

For example (case of my local machine M. Glushanina): 
```
python run_pipeline.py C:/Users/mariy/Desktop/gsoc2025/used_files/n9_listen_cleaned_manually.set C:/Users/mariy/Desktop/gsoc2025/used_files/n10_speak_cleaned_manually.set  --name1 nt9_listen --name2 nt10_speak  --montage C:/Users/mariy/Desktop/gsoc2025/GSN-HydroCel-65_1.0.sfp
```

The two files in question should be people doing the task simultaneously, e.g. NT9 speaking and NT10 listening for the case of NT9-10 pair due to preprocessing pipeline 

# Step 4.1: Beware of very noisy data 

If the data is too noisy, the algorithm will tell you. In this case mark the dataset is too noisy in our common spreadsheet 

# Step 5: Do everything the script asks you 

Meaning: 
1. Insert number of the components (for now it's 30)
2. You'll see plots popping up with ICA information. Close them, so the script could continue running 

You'll see the plots here: 

3. 