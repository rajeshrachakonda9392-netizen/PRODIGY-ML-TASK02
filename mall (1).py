
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


data = {
    "CustomerID": range(1, 61),
    "Gender": [
        "Male","Male","Female","Female","Female","Female","Female","Female","Male","Female",
        "Male","Female","Female","Female","Male","Male","Female","Male","Male","Female",
        "Male","Male","Female","Male","Female","Male","Female","Male","Female","Female",
        "Male","Female","Male","Male","Female","Female","Female","Female","Female","Female",
        "Female","Male","Male","Female","Female","Female","Female","Female","Female","Female",
        "Female","Male","Female","Male","Female","Male","Female","Male","Female","Male"
    ],
    "Age": [
        19,21,20,23,31,22,35,23,64,30,67,35,58,24,37,22,35,20,52,35,
        35,25,46,31,54,29,45,35,40,23,60,21,53,18,49,21,42,30,36,20,
        65,24,48,31,49,24,50,27,29,31,49,33,31,59,50,47,51,69,27,53
    ],
    "Annual Income (k$)": [
        15,15,16,16,17,17,18,18,19,19,19,19,20,20,20,20,21,21,23,23,
        24,24,25,25,28,28,28,28,29,29,30,30,33,33,33,33,34,34,37,37,
        38,38,39,39,39,39,40,40,40,40,42,42,43,43,43,43,44,44,46,46
    ],
    "Spending Score (1-100)": [
        39,81,6,77,40,76,6,94,3,72,14,99,15,77,13,79,35,66,29,98,
        35,73,5,73,14,82,32,61,31,87,4,73,4,92,14,81,17,73,26,75,
        35,92,36,61,28,65,55,47,42,42,52,60,54,60,45,41,50,46,51,46
    ]
}

df = pd.DataFrame(data)


df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})


X = df[["Annual Income (k$)", "Spending Score (1-100)"]]



scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


kmeans = KMeans(n_clusters=5, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Cluster"],
    cmap="viridis",
    s=100
)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation using K-Means")
plt.colorbar(label="Cluster")
plt.show()


print(df.head())
