import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")
combine = pd.concat([train.drop('target',1),test])

train.info()
train.describe()


if 1 == 0:
    # Missing Values
    missing = train.isnull().sum()
    f = open('./figures/missing_values.txt', 'w')
    f.write(str(missing))
    f.close()

    # HeatMap
    plt.figure(figsize=(140,120))
    foo = sns.heatmap(train.drop('id',axis=1).corr(), vmax=0.6, square=True, annot=True)
    plot = foo.get_figure()
    plot.savefig('./figures/heatmap.png')