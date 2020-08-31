import sys
import json
import os
from collections import defaultdict
import pandas as pd

def accumulate_pass_results(path):
    cnt = 0
    dat = []
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.startswith("pass_data"):
                p = os.path.join(dirname, filename)
                with open(p, "r") as f:
                    lines = f.read().splitlines()
                    for l in lines:
                        try:
                            a = json.loads(l)
                            # a["input"].pop("pass_history")
                            dat.append(a)
                        except Exception as e:
                            print(p, l, e)
                            cnt+=1
        print(len(dat), cnt)
    return dat

def accumulate_pass_results_direct(path):
    cnt = 0
    cnt2=0
    dfs =  []
    for dirname, _, filenames in os.walk(path):

        dat = []
        for filename in filenames:
            if filename.startswith("pass_data"):
                p = os.path.join(dirname, filename)
                with open(p, "r") as f:
                    lines = f.read().splitlines()
                    for l in lines:
                        try:
                            a = json.loads(l)
                            # a["input"].pop("pass_history")
                            dat.append(a)
                            cnt2+=1
                        except Exception as e:
                            print(p, l, e)
                            cnt+=1
        dfs.append(to_dataframe(dat))
        print(cnt2, cnt)
    return pd.concat(dfs)


def to_dataframe(dat):
    def f(x):
        y = x["input"]["feature"]
        y["pass"] = x["input"]["pass"]
        y["name"] = x["IR_name"]
        y["modified"] = x["modified"]
        return y
    dat = pd.DataFrame(list(map(f, dat)))
    return dat 


if __name__=="__main__":
    data = accumulate_pass_results_direct(sys.argv[1])
    # data = to_dataframe(data)
    data.to_csv("data.csv")
