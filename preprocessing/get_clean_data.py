import numpy as np
from clean_text import clean_text
import csv
import sys
import time
from os import path
from paths import DATA_PATH, OUT_PATH
csv.field_size_limit(sys.maxsize)

def read_data(year):
    f = open(DATA_PATH+str(year)+'.csv', 'r')
    data = list(csv.reader(f, quoting=csv.QUOTE_NONE, delimiter=';'))
    f.close()
    print('Found ', len(data), "entries.")
    return data

def read_classification_data(data_path, remove_punctuation_marks=True, remove_stopwords=True, multiword=False):
    p = data_path.rpartition('.')
    clean_data_path = p[0]+'_cleaned.'+p[2]
    if(not path.exists(p[0]+'_cleaned.'+p[2])):
        print('no cleaned data found for: '+ data_path)
        f = open(data_path, 'r')
        data = list(csv.reader(f, quoting=csv.QUOTE_NONE, delimiter=';'))
        f.close()
        print('Found ', len(data), "entries.")
        f=open(clean_data_path,'a', errors='ignore')
        prev = time.time()
        entries = 0
        buffers = []
        for i in range(0, len(data)):
            orig_text = data[i][1].replace('"', '').replace("'", '')
            clean_entry, _ = clean_text(orig_text,  clean_only=True, remove_stopwords=remove_stopwords, remove_punctuation_marks=remove_punctuation_marks)
            print(i,"/" ,entries," took: ",time.time() - prev, " len tokens: ", len(clean_entry))
            prev = time.time()
            if(len(clean_entry) > 1):
                entries += 1
                buffers.append(clean_entry)
                if((len(buffers)+1) % 3 == 0):
                    np.savetxt(f, buffers, fmt='%s', delimiter=",")
                    buffers = []
        f.close()
    else:
        print('clean data found at: '+ clean_data_path)
        entries = []
        for year in range(2017,2018):
            if(not multiword):
                f = open(clean_data_path, 'r')
                f = (l.replace('\0', '') for l in f)
                entries = entries + list(csv.reader(f, quotechar="'"))
                for i,data in enumerate(entries):
                    entries[i] = [d.replace("'", '').replace("[","").replace("]","").strip() for d in data]
                f.close()
        return entries

def read_clean_data_for_year(year, remove_punctuation_marks, remove_stopwords, multiword=False):
    entries = []    
    data_path = get_path(remove_punctuation_marks, remove_stopwords)
    if(not multiword):
        if(path.exists(data_path+str(year)+'.csv')):
            f = open(data_path+str(year)+'.csv', 'r')
            f = (l.replace('\0', '') for l in f)
            entries = entries + list(csv.reader(f, quotechar="'"))
            for i,data in enumerate(entries):
                entries[i] = [d.replace("'", '').replace("[","").replace("]","").strip() for d in data]
            f.close()
        else:
            print(data_path+str(year)+'.csv not found')
    else:
        if(path.exists(data_path+str(year)+'_mw.csv')):
            f = open(data_path+str(year)+'_mw.csv', 'r')
            f = (l.replace('\0', '') for l in f)
            entries = entries + list(csv.reader(f, quotechar="'"))
            for i,data in enumerate(entries):
                entries[i] = [d.replace("'", '').replace("[","").replace("]","").strip() for d in data]
            f.close()
        else:
            print(data_path+str(year)+'_mw.csv not found')
    return entries

def read_cleaned_data(remove_punctuation_marks, remove_stopwords, multiword=False): 
    entries = []
    data_path = get_path(remove_punctuation_marks, remove_stopwords) 
    for year in range(2016,2017):
        entries = read_clean_data_for_year(year, remove_punctuation_marks, remove_stopwords, multiword)
    print('Found ', len(entries), "entries.")
    return entries

def clean_data(remove_punctuation_marks, remove_stopwords):
    start = time.time()
    data_path = get_path(remove_punctuation_marks, remove_stopwords)  
    for year in range(2018,2019):
        data = read_data(year)
        entries = 0

        buffers = []
        mw_buffer = []
        f=open(data_path+str(year)+'.csv','a', errors='ignore')
        prev = start
        print('cleaning ', year)
        for i in range(0, len(data)): 
            orig_text = ' '.join(data[i][2:]).replace('"', '').replace("'", '')
            if(orig_text.strip() != '' ):
                clean_entry, _ = clean_text(orig_text, clean_only=True, remove_stopwords=remove_stopwords, remove_punctuation_marks=remove_punctuation_marks)
                print(i,"/" ,entries," took: ",time.time() - prev, " len tokens: ", len(clean_entry))
                prev = time.time()
                if(len(clean_entry) > 1):
                    entries += 1
                    buffers.append(clean_entry)
                    if((len(buffers)+1) % 3 == 0):
                        np.savetxt(f, buffers, fmt='%s', delimiter=",")
                        buffers = []
            else:
                print(orig_text)
        f.close()
        end = time.time()
        print('took overall: ', end - start)
    return entries

def get_multiword_text(tokens):
    from multiword import is_multiword
    mw_tokens = []
    i = 0
    while i < len(tokens)-1:
        token = tokens[i]
        following = tokens[i+1]
        if(is_multiword(token, following)):
            mw_tokens.append(token+'_'+following)
            i += 1
        else:
            mw_tokens.append(token)
        i+=1
    return mw_tokens


def get_path(remove_punctuation_marks, remove_stopwords):
    data_path = OUT_PATH 
    data_path += "no_pm" if(remove_punctuation_marks) else 'pm'
    data_path += '_no_sw' if(remove_stopwords) else '_sw'
    return data_path + '/clean_entries_'

def get_clean_data(clean=False, remove_punctuation_marks=False, remove_stopwords=False, multiword=False):  
    data_path = get_path(remove_punctuation_marks, remove_stopwords)  
    if(not path.exists(data_path+str(2017)+'.csv') or clean):
        print(data_path+str(2017)+'.csv not found. cleaning data...')
        entries = clean_data(remove_punctuation_marks, remove_stopwords)
    else:
        print('clean data found at: ' + data_path+str(2017)+'.csv')
        entries = read_cleaned_data(remove_punctuation_marks, remove_stopwords, multiword)
    return entries

if __name__ == '__main__':
    get_clean_data(clean=True, remove_punctuation_marks=False, remove_stopwords=False)