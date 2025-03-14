from sklearn.decomposition import PCA, KernelPCA
from sklearn.cross_decomposition import CCA
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from copy import deepcopy
from xlrd import open_workbook
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import defaultdict
from mtools import sortNondominated
import time
import joblib
from itertools import combinations

data=pd.read_excel('coaldata.xlsx', sheet_name='Sayfa1')
col_names=[]
for col in data.columns:
    col_names.append(col)

ranks=data.get(col_names[3])
out_data = [0]*21
for i in range(4,25):
    if i in [4,9,11,12,13,14,15,16,17,19]:
        out_data[i-4]=np.array(data.get(col_names[i]))*(-1)
    else:
        out_data[i-4]=np.array(data.get(col_names[i]))
# Now the output is permitted between out_data[0] to out_data[20] columns.

# Preprocessing, standard scaling the data.
data_2d_array = np.vstack(out_data)
data_2d_array = data_2d_array.T

std_scaler = MinMaxScaler()
scaled_2d_data = std_scaler.fit_transform(data_2d_array)

# Making a pd dataframe again
df_index=[]
for i in range(len(ranks)):
    df_index.append(str(i))

y_col_names=[]
for i in range(len(col_names)-4):
    y_col_names.append(col_names[i+4])

original_df = pd.DataFrame(data = scaled_2d_data,  
                  index = df_index,  
                  columns = y_col_names) 

original_df = original_df.drop(original_df.columns[11], axis=1)

# Split the DataFrame into two 265x10 DataFrames
df_X = original_df.iloc[:, :10]
df_Y = original_df.iloc[:, 10:]

# Apply Canonical Correlation Analysis (CCA)
cca = CCA(n_components=10)  # You can adjust the number of components based on your requirements
cca_result = cca.fit_transform(df_X, df_Y)

my_array = np.array(cca_result[0])

# Split the array into two arrays along the second axis (axis=1)
array1, array2 = np.split(my_array, 2, axis=1)

cca_2 = CCA(n_components=5)  # You can adjust the number of components based on your requirements
cca_2_result = cca_2.fit_transform(array1, array2)

cca_2_x_results = cca_2_result[0]

def fitness(pop_index,comb):
    result=[]
    for i in range(len(comb)):
        result.append(cca_2_x_results[pop_index,comb[i]])
    return result

def comb_to_pareto(comb):
    pop=defaultdict(list)
    for index in range(265):
        pop[index].append(index)
        pop[index].append(0)
    for key in pop:
        fit_res=fitness(key,comb)
        pop[key].append(fit_res)
    pareto_front = sortNondominated(pop, len(pop),first_front_only=True)
    return pareto_front
    
def try_add_sorting(array,individual):
    #try adding individual to an existing sorting list [[index,value],[index,value]]
    for i in range (len(array)):
        if individual[1] > array[i][1]:
            for j in range(len(array)-i):
                if j!=0:
                    array[len(array)-j]=deepcopy(array[len(array)-1-j])
            array[i]=deepcopy(individual)
            break

out_data_indices=[i for i in range(0,len(cca_2_x_results[0]))]
master_distrib=[0]*266
master_top_10=[[0,0] for i in range(10)]
t_init= time.time()
list_0_to_265=[i for i in range(266)]

file_name="CCA_recorder.txt"

for i in range(0,9): #len(out_data_indices)
    comb_M=i+1
    combs = combinations(out_data_indices,comb_M)
    comb_list = []
    distribution=[0]*266
    top_10=[[0,0] for i in range(10)]
    iter_counter=0

    for j in combs:
        comb_list.append(list(j))
        
    with joblib.Parallel(n_jobs=8) as parallel:
        pareto_fronts=parallel(joblib.delayed(comb_to_pareto)(comb) for comb in comb_list)

    for pareto_front in pareto_fronts:
        for front_elements in range(len(pareto_front[0])):
            rank_num=pareto_front[0][front_elements][0]
            distribution[ranks[rank_num]]+=1
    
    line="The "+str(i+1)+" combinations result:\n"
    # backup the distrib in a text file,
    for j in range(len(distribution)):
        line=line+str(distribution[j])+" "
    line=line+"\n"
    with open(file_name,"a") as opened_file:
        opened_file.write(line)

    print("Combs of",i+1,"is complete.")
    t_cur= time.time()
    print("Elapsed time:",(t_cur-t_init)/60,"mins")
    for index in range(len(distribution)):
        ind=[index,distribution[index]]
        try_add_sorting(top_10,ind)
    if top_10[0][1]!=top_10[-1][1]:
        for ind in top_10:
            master_distrib[ind[0]]+=1

for index in range(len(master_distrib)):
    ind=[index,master_distrib[index]]
    try_add_sorting(master_top_10,ind)

line="Final result:\n"
for j in range(len(master_top_10)):
    line=line+str(master_top_10[j])+" "
with open(file_name,"a") as opened_file:
    opened_file.write(line)
    line=line+"\n"

print("Top 10 of all possible combinations:",master_top_10)
master_distrib[0]=0
plt.plot(list_0_to_265,master_distrib)
plt.xlabel("Location Rank Indicator")
plt.ylabel("Number of being in top 10 in existing objective combinations")
plt.grid()
plt.show()