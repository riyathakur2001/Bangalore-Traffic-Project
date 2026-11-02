
import os
# These two lines stop a black cmd window from opening during K-Means (Windows only)
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
os.environ["OMP_NUM_THREADS"] = "1"
import warnings
warnings.filterwarnings("ignore")





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import KMeans

# ---------- STEP 1: Load the data ----------
df = pd.read_csv(r"C:\Users\riyat\OneDrive\Desktop\advanced python\bangalore_traffic_real.csv")
print("STEP 1: Data loaded")
print("Rows:", len(df), "| Junctions:", df["location"].nunique())

# ---------- STEP 2: Clean the data ----------
df = df.drop_duplicates()                 # remove repeated rows
df = df.dropna()                          # remove empty values
df = df[df["free_flow_speed"] > 0]        # speed must be positive
print("\nSTEP 2: After cleaning, rows =", len(df))

# ---------- STEP 3: Make new columns (feature engineering) ----------
# Congestion ratio = current speed / normal (free flow) speed
# ratio = 1 means no jam, small ratio means heavy jam
df["ratio"] = df["current_speed"] / df["free_flow_speed"]
df["ratio"] = df["ratio"].clip(upper=1)

# Divide traffic in 3 levels
def make_level(r):
    if r >= 0.85:
        return "Low"
    elif r >= 0.65:
        return "Medium"
    else:
        return "High"

df["level"] = df["ratio"].apply(make_level)

# Hour is a circle (23 is close to 0), so we use sin and cos of the hour
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["is_weekend"] = (df["weekday"] >= 5).astype(int)

print("\nSTEP 3: Traffic levels")
print(df["level"].value_counts())

# ---------- STEP 4: Explore the data (EDA) ----------
print("\nSTEP 4: Exploring the data")
by_hour = df.groupby("hour")["ratio"].mean()
peak_hour = by_hour.idxmin()              # lowest ratio = most jam
print("Most jammed hour of the day:", peak_hour, ":00")

by_junction = df.groupby("location")["ratio"].mean().sort_values()
print("\nTop 5 most congested junctions:")
print(by_junction.head(5).round(3))

# ----- GRAPH 1: Traffic pulse of the city (average ratio by hour) -----
plt.figure("Graph 1 - Traffic Pulse", figsize=(9, 5))
plt.plot(by_hour.index, by_hour.values, color="#444444", zorder=1)
dot_colors = ["#2a9d8f" if r >= 0.85 else ("#f4a261" if r >= 0.65 else "#e63946") for r in by_hour.values]
plt.scatter(by_hour.index, by_hour.values, c=dot_colors, s=90, zorder=2)
plt.axhline(0.85, color="gray", linestyle="--", linewidth=1)
plt.text(0, 0.86, "above this line = Low traffic", fontsize=8, color="gray")
plt.annotate("Worst hour: " + str(peak_hour) + ":00", xy=(peak_hour, by_hour.min()),
             xytext=(peak_hour - 8, by_hour.min() - 0.02), arrowprops=dict(arrowstyle="->", color="red"),
             color="red", fontweight="bold")
plt.title("Traffic Pulse of Bangalore: how congestion changes through the day")
plt.xlabel("Hour of the day"); plt.ylabel("Average congestion ratio (lower = more jam)")
plt.xticks(range(0, 24, 2)); plt.grid(alpha=0.3)
plt.ylim(by_hour.min() - 0.06, 1.03)
plt.tight_layout()
plt.savefig("graph1_traffic_pulse.png")
plt.show()          # window opens; close it to continue

# ----- GRAPH 2: Heat map - which junction is jammed at which time of day -----
df["part_of_day"] = pd.cut(df["hour"], [-1, 5, 11, 16, 21, 23],
                           labels=["Night", "Morning", "Afternoon", "Evening", "Late night"])
heat = df.pivot_table(index="location", columns="part_of_day", values="ratio", aggfunc="mean")
heat = heat.loc[by_junction.index]            # most congested junction on top
plt.figure("Graph 2 - Junction Heat Map", figsize=(8, 8))
sns.heatmap(heat, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0.5, vmax=1.0,
            cbar_kws={"label": "Congestion ratio (red = jam, green = free)"})
plt.title("Where and when is Bangalore jammed?")
plt.xlabel("Part of the day"); plt.ylabel("")
plt.tight_layout()
plt.savefig("graph2_junction_heatmap.png")
plt.show()          # window opens; close it to continue

# ---------- STEP 5: Regression (predict the ratio, a number) ----------
features = ["hour_sin", "hour_cos", "weekday", "is_weekend", "lat", "lon", "free_flow_speed"]
X = df[features]          # inputs
y = df["ratio"]           # what we want to predict

# 80% data for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nSTEP 5: Regression models")
models = {"Linear Regression": LinearRegression(),
          "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42)}
best_r2 = -1
for name in models:
    model = models[name]
    model.fit(X_train, y_train)               # learn from training data
    pred = model.predict(X_test)              # predict on unseen test data
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    print(name, "-> MAE:", round(mae, 3), "| RMSE:", round(rmse, 3), "| R2:", round(r2, 3))
    if r2 > best_r2:
        best_r2, best_name, best_pred = r2, name, pred

# ----- GRAPH 3: actual vs predicted (how good is our model?) -----
plt.figure("Graph 3 - Model Check", figsize=(6.5, 6))
true_levels = y_test.apply(make_level)
for lvl, col in [("Low", "#2a9d8f"), ("Medium", "#f4a261"), ("High", "#e63946")]:
    mask = (true_levels == lvl).values
    plt.scatter(y_test[mask], best_pred[mask], c=col, alpha=0.6, label=lvl + " traffic")
plt.plot([0.3, 1], [0.3, 1], "k--", label="perfect prediction")
plt.text(0.32, 0.93, best_name + "\nR2 = " + str(round(best_r2, 3)), fontsize=10,
         bbox=dict(boxstyle="round", facecolor="white"))
plt.xlabel("Actual congestion ratio"); plt.ylabel("Predicted congestion ratio")
plt.title("Can the model predict traffic?")
plt.legend(loc="lower right"); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("graph3_model_check.png")
plt.show()          # window opens; close it to continue


# ----- GRAPH 4: What drives traffic? (feature importance of Random Forest) -----
rf_model = models["Random Forest"]
importance = pd.Series(rf_model.feature_importances_, index=features).sort_values()
plt.figure("Graph 4 - What drives traffic", figsize=(8, 5))
importance.plot(kind="barh", color="#3a86ff")
plt.title("What decides the traffic? (Random Forest feature importance)")
plt.xlabel("Importance (bigger = more useful for prediction)")
plt.tight_layout()
plt.savefig("graph4_feature_importance.png")
plt.show()          # window opens; close it to continue

# ---------- STEP 6: Classification (predict Low / Medium / High) ----------
print("\nSTEP 6: Classification")
yc = df["level"]
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X, yc, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(Xc_train, yc_train)
pred_c = clf.predict(Xc_test)
accuracy = accuracy_score(yc_test, pred_c)
print("Accuracy:", round(accuracy * 100, 1), "%")
print("Confusion matrix (rows = actual, columns = predicted, order Low, Medium, High):")
print(confusion_matrix(yc_test, pred_c, labels=["Low", "Medium", "High"]))

# ----- GRAPH 5: Confusion matrix as a picture -----
cm = confusion_matrix(yc_test, pred_c, labels=["Low", "Medium", "High"])
plt.figure("Graph 5 - Confusion Matrix", figsize=(5.5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Low", "Medium", "High"], yticklabels=["Low", "Medium", "High"])
plt.xlabel("Predicted level"); plt.ylabel("Actual level")
plt.title("Classification: how many did we get right?")
plt.tight_layout()
plt.savefig("graph5_confusion_matrix.png")
plt.show()          # window opens; close it to continue

# ---------- STEP 7: Clustering (group similar junctions) ----------
print("\nSTEP 7: K-Means clustering of junctions")
profile = df.groupby("location")[["ratio"]].mean()
profile["night_ratio"] = df[df["hour"] <= 3].groupby("location")["ratio"].mean()
profile["evening_ratio"] = df[(df["hour"] >= 19) & (df["hour"] <= 21)].groupby("location")["ratio"].mean()
profile = profile.fillna(1)
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
profile["group"] = kmeans.fit_predict(profile[["ratio", "night_ratio", "evening_ratio"]])
for g in range(3):
    members = list(profile[profile["group"] == g].index)
    print("Group", g + 1, "(avg ratio", round(profile[profile["group"] == g]["ratio"].mean(), 2), "):", ", ".join(members))

# ----- GRAPH 6: Junction groups found by K-Means -----
plt.figure("Graph 6 - Junction Groups", figsize=(8, 6))
group_colors = ["#2a9d8f", "#e63946", "#f4a261"]
for g in range(3):
    part = profile[profile["group"] == g]
    plt.scatter(part["ratio"], part["evening_ratio"], s=90, c=group_colors[g], label="Group " + str(g + 1))
for name in profile.index:
    plt.annotate(name, (profile.loc[name, "ratio"], profile.loc[name, "evening_ratio"]),
                 fontsize=7, xytext=(4, 4), textcoords="offset points")
plt.xlabel("Average congestion ratio (whole data)"); plt.ylabel("Evening congestion ratio (7-9 PM)")
plt.title("K-Means: junctions with similar traffic behaviour")
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("graph6_junction_groups.png")
plt.show()          # window opens; close it to continue

# ---------- STEP 8: CONCLUSION ----------
print("\n==================== CONCLUSION ====================")
print("1.", len(df), "real traffic readings of", df["location"].nunique(), "Bangalore junctions were studied.")
print("2. The most jammed time is around", peak_hour, ":00 and the most congested junction is", by_junction.index[0], ".")
print("3. The best regression model is", best_name, "with R2 =", round(best_r2, 3), ".")
print("4. The classifier can tell Low/Medium/High traffic with", round(accuracy * 100, 1), "% accuracy.")
print("5. Time of the day is the main reason for traffic changes; location comes next.")
print("6. Limitation: data is only of a few days, so results are approximate.")
