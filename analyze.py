import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
# from sklearn.svm import LinearSVC


filename = "SpotifyFeatures.csv"

metas = ["popularity", "duration_ms", "tempo", "key",
         "mode", "time_signature", "tempo", "loudness"]

features = ["acousticness", "danceability", "energy",
            "instrumentalness", "liveness", "speechiness", "valence"]


tracks = pd.read_csv(filename)
print(tracks.describe())


# --- FEATURE CORRELATION ----


correlation = tracks.corr(method='spearman')
print(correlation)

plt.figure(figsize=(20, 10))
plt.title('Correlation heatmap of Spotify features')
sns.heatmap(correlation, vmin=-1, vmax=1, cmap="bwr")
# plt.show()
plt.savefig("output/correlation")
plt.close()


# --- ACOUSTICNESS VS ENERGY ---

plt.title("Acousticness vs Energy")
plt.scatter(tracks["acousticness"], tracks["energy"], alpha=0.005)
plt.xlabel("Acousticness")
plt.ylabel("Energy")
# plt.show()
plt.savefig("output/acousticness_vs_energy")
plt.close()


# --- ACOUSTICNESS VS LOUDNESS ---


plt.title("Acousticness vs Loudness")
plt.scatter(tracks["acousticness"], tracks["loudness"], alpha=0.005)
plt.xlabel("Acousticness")
plt.ylabel("Loudness")
# plt.show()
plt.savefig("output/acousticness_vs_loudness")
plt.close()


# --- DANCEABILITY VS VALENCE ---

plt.title("Danceability vs Valence")
plt.scatter(tracks["danceability"], tracks["valence"], alpha=0.005)
plt.xlabel("Danceability")
plt.ylabel("Valence")
# plt.show()
plt.savefig("output/danceability_vs_valence")
plt.close()


# --- MODE ANALYSIS ---


x1 = tracks.loc[np.where(tracks["mode"] == "Major")]
x2 = tracks.loc[np.where(tracks["mode"] == "Minor")]

kwargs = dict(alpha=1, bins=100)

for f in features + metas:
    plt.hist(x1[f], **kwargs, color='dodgerblue', label='Major')
    plt.hist(x2[f], **kwargs, color='orange', label='Minor')
    plt.gca().set(title='Mode vs ' + f, xlabel=f, ylabel="Nb of tracks")
    plt.legend()
    # plt.show()
    plt.savefig("output/mode_vs/" + f)
    plt.close()


# --- GENRE FEATURE RADAR ---


genre_names = tracks.genre.unique()
genre_meds = []

for g in genre_names:
    subtracks = tracks.loc[np.where(tracks["genre"] == g)]
    genre_meds.append([np.median(subtracks[f]) for f in features])

angles = np.linspace(0, 2*np.pi, len(features),
                     endpoint=False)  # Set the radar angles
angles = np.concatenate((angles, [angles[0]]))

for i in range(len(genre_names)):
    name = genre_names[i]
    med = genre_meds[i]
    med = np.concatenate((med, [med[0]]))  # Closed
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)   # Set polar axis
    ax.plot(angles, med, '-', linewidth=2)
    ax.fill(angles, med, alpha=0.25)  # Fulfill the area
    # Set the label for each axis
    ax.set_thetagrids(angles * 180/np.pi, features)
    ax.set_title(name)
    ax.set_rlim(0, 1)
    ax.grid(True)
    plt.savefig("output/radar_genres/" + name)
    # plt.show()
    plt.close()


# --- REGRESSION - Train & validation dataset ---


# Mapping all features to popularity
x = tracks.loc[:, features].values
y = tracks.loc[:, 'popularity'].values

# Creating a test and training dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)


# --- REGRESSION - Linear Regression ---


regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


# --- REGRESSION - Random Forest Classifier ---


# clf = RandomForestClassifier(n_estimators=10)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)


# --- REGRESSION - Linear SVM ---


# clf = LinearSVC(penalty='l2', loss='squared_hinge', dual=True)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)


# --- REGRESSION - Accuracy measurement ---


df_output = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df_output)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(
    metrics.mean_squared_error(y_test, y_pred)))
