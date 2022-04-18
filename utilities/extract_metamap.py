# path: path of a metamap output file
# semantic_types: a set of given UMLS semantic types
# mf: dictionary of all unique metamap features belong to the given semantic types and the count of such terms are also recorded 

import codecs, re

def feature_generation_from_metamap_output(path,semantic_types): 
    note = codecs.open(path,encoding='utf-8',mode='r').readlines()
    mf={}
    for l in range(0,len(note)): 
        line=note[l].strip('\n')
        result=re.search( r'(.+?)(\b[7-9][0-9][0-9]|1000\b)([ ]+)([C])([0-9]+)(:)(.+)', line)  # change the confidence score  here from 1000 to any other number, if required
        if result:
            match=result.group(0)                    
            st=re.search( r'(\[)([a-zA-Z, ]+)(\])', match)
            if st and semantic_types.count(st.group(2)):      # Checking the desired semantic type
                ptrn=re.search( r'([C])([0-9]+)(:)', match)   # Extracting concept id
                if ptrn:
                    cid=ptrn.group(1)+ptrn.group(2)                                                  
                    if cid not in list(mf.keys):
                        mf[cid]=[]
                        mf[cid]=1
                        print(cid+'\t'+st.group(2))
                    else:
                        mf[cid]+=1
    return mf 