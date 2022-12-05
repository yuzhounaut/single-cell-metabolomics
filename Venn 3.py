import pandas as pd
from matplotlib_venn import venn2, venn3
import matplotlib.pyplot as plt
import os,sys
os.chdir(sys.path[0])

from matplotlib import rcParams
# figure size in inches
rcParams['figure.figsize'] = 6,6
rcParams['figure.dpi'] = 300

path = r"D:\PyProject\Outlier\Hela-Noco-Starv-ML-Venn.xlsx"
def excel_one_line_to_list(path, colnumber):
    df = pd.read_excel(path, usecols=[colnumber], names=None)  
    df_li = df.values.tolist()
    result = []
    for s_li in df_li:
        result.append(s_li[0])
    return result

data = pd.read_excel(path, sheet_name=0, nrows=1)
col_header = data.columns
first_col = col_header[0]
second_col = col_header[1]
third_col = col_header[2]
CatBoost  = excel_one_line_to_list(path, 0)
XGBoost  = excel_one_line_to_list(path, 1)
LightGBM  = excel_one_line_to_list(path, 2)

subset3 = [set(CatBoost), set(XGBoost), set(LightGBM)]

A = set(CatBoost)
B = set(XGBoost)
C = set(LightGBM)
total = len(A.union(B, C))

plt.figure(dpi=300)
v = venn3(subset3, set_labels=(first_col,second_col,third_col), set_colors=('#ffcc00','#189fdd','#ef4927'), 
          alpha=0.8, ax=None, subset_label_formatter=lambda x: str(x) + "\n(" + f"{(x/total):1.0%}" + ")")
# subset_label_formatter=lambda x: f"{(x/total):1.0%}"
for text in v.set_labels:
    text.set_fontsize(14)
for text in v.subset_labels:
    text.set_fontsize(16)
#plt.savefig("daasshuzi.tif", dpi= 300)
plt.show()
