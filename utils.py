import os
import json 
import pickle

def get_test_data(confidence_score,tst_path,metamap = False):
    print(tst_path)
    if confidence_score>0.85:
        print('\n ***** Getting Test Data ***** \n')   
        tst_dict={}; tst_data=[]; tst_vectors = [] 
        unique_id=[]; 
        tst_files=os.listdir(tst_path)
        vec = {}
        if metamap:
            f = open('metamap_vectors_test.pkl','rb')
            vec = pickle.load(f)
        if tst_files==[]:
                print('There is no test document in the directory \n')
        else:
            for elm in tst_files:
                if elm.find('.json')>0:                             # Checking if it is a JSON file
                    fl=open(tst_path+elm, 'r')  
                    reader = json.load(fl)
                    fl.close()        
                    for item in reader:
                        idn=item['nick']
                        if item['number']>=0 and idn not in unique_id:
                            unique_id.append(idn)
                            tst_dict[idn]=[]
                            tst_dict[idn].append(item['content'])
                        elif idn in unique_id:
                            tst_dict[idn].append(item['content'])
                    for item in tst_dict:
                        text=''.join(tst_dict[item])
                        tst_data.append(text)
                        if metamap:
                            tst_vectors.append(vec[item])
        return tst_data,tst_dict,tst_vectors
            