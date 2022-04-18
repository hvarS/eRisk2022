import glob 

folders = ['2018_cases','2017_cases']
labels = ['pos','neg']

for folder in folders:
    for label in labels:
        for file in glob.glob(f'task2_data/t2_training/TRAINING_DATA/{folder}/{label}/*'):
            with open('risk_golden_truth.txt','a') as f:
                if label == 'pos':
                    f.write((file.split('/')[-1]).split('.')[0]+" 1\n")
                else:
                    f.write((file.split('/')[-1]).split('.')[0]+" 0\n")
