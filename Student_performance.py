import pandas as pd
import numpy as np
import plotly.express
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.neighbors import LocalOutlierFactor
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering

# ayarlar
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 1000)

# veri setini okuma
df = pd.read_csv("Dataset/Student_Performance.csv")
df.head()


# veri seti ile ilgili genel resim (EDA);
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)

# Kullanmayacağımız değişkeni veri setinden çıkartacağız.

df.drop("Unnamed: 0", axis=1, inplace=True)


# Veride ki numerik ve kategorik değişkenlerini ayrıştırmak istersek;
def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car, num_but_cat


cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)


# kategorik verilerin sınıflarını gözlemlemek istersek;
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col, plot=True)


def pie_plt(dataframe, col_name):
    dataframe[col_name].value_counts().plot.pie(autopct='%1.1f%%', figsize=(10, 8))
    plt.show()
    return


for col in cat_cols:
    pie_plt(df, col)


# Numerik verilerin analizi;


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


num_summary(df, num_cols, plot=True)

# cinsiyete göre pairplot grafikleri;
sns.pairplot(df, hue="Gender")
plt.show()

# Sınava hazırlık kursu alıp almadığına göre pairplot grafikleri;
sns.pairplot(df, hue="TestPrep")
plt.show()

# etnik gruba göre pairplot grafikleri;
sns.pairplot(df, hue="EthnicGroup")
plt.show()

# ebeveyn eğitim durumuna göre pairplot grafiği;
sns.pairplot(df, hue="ParentEduc")
plt.show()

# Korelasyon'a bakmak istersek;

cor = df.corr()

sns.heatmap(cor, annot=True)
plt.show()

# Farklı feature'lara göre numerik değerlerin betimsel istatistiklerine bakmak istersek;

df.groupby(["Gender", "EthnicGroup", "ParentEduc"]).agg({"MathScore": ["mean", "std", "min", "max"],
                                                         "ReadingScore": ["mean", "std", "min", "max"],
                                                         "WritingScore": ["mean", "std", "min", "max"]})

# Nümerik Değişkenlerin normal dağılımına bakma ve standartlaştırma işlemi;

model_df = df[["MathScore", "ReadingScore", "WritingScore"]]
model_df.head()


def check_skew(df_skew, column):
    skew = stats.skew(df_skew[column])
    skewtest = stats.skewtest(df_skew[column])
    plt.title('Distribution of ' + column)
    sns.distplot(df_skew[column], color="g")
    print("{}'s: Skew: {}, : {}".format(column, skew, skewtest))
    return


plt.figure(figsize=(9, 9))
plt.subplot(6, 1, 1)
check_skew(model_df, 'MathScore')
plt.subplot(6, 1, 2)
check_skew(model_df, 'ReadingScore')
plt.subplot(6, 1, 2)
check_skew(model_df, 'WritingScore')
plt.subplot(6, 1, 2)
plt.savefig('before_transform.png', format='png', dpi=1000)
plt.show()

# Normal dağılımın sağlanması için Log transfarmotion'ın uygulanması;

model_df["MathScore"] = np.log1p(model_df["MathScore"])
model_df["ReadingScore"] = np.log1p(model_df["ReadingScore"])
model_df["WritingScore"] = np.log1p(model_df["WritingScore"])
model_df.head()

# Scaling işlemi

sc = MinMaxScaler((0, 1))
model_scaling = sc.fit_transform(model_df)
model_df = pd.DataFrame(model_scaling, columns=model_df.columns)
model_df.head()

# K Means Yöntemi için küme sayısı belirlemek istersek;

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(model_df)
elbow.show()

# Modeli oluşturup öğrencileri segmentlere ayırmak istersek;

k_means = KMeans(n_clusters=6, random_state=42).fit(model_df)
segments = k_means.labels_
segments

final_df = df[
    ["Gender", "EthnicGroup", "ParentEduc", "LunchType", "TestPrep", "MathScore", "ReadingScore", "WritingScore"]]
final_df["SegmentKM"] = segments
final_df.head()

# Segmentleri istatistiksel olarak incelemek istersek;

final_df.groupby("SegmentKM").agg({"MathScore": ["mean", "min", "max"],
                                   "ReadingScore": ["mean", "min", "max"],
                                   "WritingScore": ["mean", "min", "max"],
                                   "TestPrep": ["count"],
                                   "LunchType": ["count"]})

# Hierarchical Clustering İle Öğrencileri Segmentlere ayırmak istersek;

hc_complete = linkage(model_df, 'complete')

plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_complete,
                  truncate_mode="lastp",
                  p=10,
                  show_contracted=True,
                  leaf_font_size=10)
plt.axhline(y=0.5, color='r', linestyle='--')
plt.show()

# Modeli oluşturup öğrencilerimizi segmentlere ayırmak istersek;

hc = AgglomerativeClustering(n_clusters=6)
segments1 = hc.fit_predict(model_df)
segments1

final_df["SegmentHC"] = segments1
final_df.head()

# Segmentleri istatistiksel olarak incelemek istersek;

final_df.groupby("SegmentHC").agg({"MathScore": ["mean", "min", "max"],
                                   "ReadingScore": ["mean", "min", "max"],
                                   "WritingScore": ["mean", "min", "max"],
                                   "TestPrep": ["count"],
                                   "LunchType": ["count"]})

final_df.head()

