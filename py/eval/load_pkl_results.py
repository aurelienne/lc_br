import pickle
import sys
import pandas as pd

pickle_file = sys.argv[1]
out_file = sys.argv[2]

def load_pickle(filename, column_label):
    f = open(filename, 'rb')
    data = pickle.load(f)
    #for key in data:
        #print(keys)
        #print(data[keys])
    df = pd.DataFrame.from_dict(data, orient='index', columns=column_label)
    #print(df)
    return df

df1 = load_pickle(pickle_file, ['GLMSUM_JScale'])
#df2 = load_pickle('/home/ajorge/output/2021_month01/eval_results_sumglm_AScaler.pkl', ['GLMSUM_AScale'])
#df3 = load_pickle('/home/ajorge/output/2021_month01/eval_results_maxglm_JScaler.pkl', ['GLMMAX_JScale'])
#df4 = load_pickle('/home/ajorge/output/2021_month01/eval_results_maxglm_AScaler.pkl', ['GLMMAX_AScale'])
#dfm1 = df1.merge(df2, left_index=True, right_index=True)
#dfm2 = dfm1.merge(df3, left_index=True, right_index=True)
#dfm3 = dfm2.merge(df4, left_index=True, right_index=True)
#dfm3.to_csv('/home/ajorge/output/2021_month01/results.csv')
df1.to_csv('/home/ajorge/lc_br/data/results/eval/' + out_file)
