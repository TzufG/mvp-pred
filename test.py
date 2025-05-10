# ===========================================================
#   NBA MVP Analytics 2012-2025  –  Feature-Selection Pipelines
# ===========================================================
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from collections import defaultdict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import make_scorer
from scipy.stats import randint, uniform
import warnings, json, pathlib
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.model_selection import cross_validate
warnings.filterwarnings("ignore")

# -----------------------------------------------------------
# 0. URLs
# -----------------------------------------------------------
urls = {yr: f"https://www.basketball-reference.com/leagues/NBA_{yr}_totals.html"
        for yr in range(2012, 2026)}

mvp_dict = {
    2012:"LeBron James", 2013:"LeBron James", 2014:"Kevin Durant",
    2015:"Stephen Curry", 2016:"Stephen Curry", 2017:"Russell Westbrook",
    2018:"James Harden",  2019:"Giannis Antetokounmpo",
    2020:"Giannis Antetokounmpo", 2021:"Nikola Jokić",
    2022:"Nikola Jokić",  2023:"Joel Embiid",  2024:"Nikola Jokić"
}

# -----------------------------------------------------------
# 1. LOAD & PREP
# -----------------------------------------------------------
dfs_prepared = {}
for yr, url in urls.items():
    df = pd.read_html(url)[0]
    df = df[df["Rk"] != "Rk"].dropna(subset=["Player"]).copy()
    df["Season"] = yr
    df["MVP"]    = (df["Player"] == mvp_dict.get(yr,"")).astype(int)
    numeric_cols = df.select_dtypes(exclude="object").columns
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    dfs_prepared[f"df_{yr}"] = df

def add_features(df):
    df = df.copy()
    df["PTS_per_game"] = df["PTS"] / df["G"].replace(0, np.nan)
    df["AST_TOV"]      = df["AST"] / (df["TOV"] + 1)
    df["REB"]          = df["ORB"] + df["DRB"]
    return df

for k in dfs_prepared:
    dfs_prepared[k] = add_features(dfs_prepared[k])

sample_df   = list(dfs_prepared.values())[0]
feature_cols = [c for c in sample_df.select_dtypes("number").columns
                if c not in ("Season","MVP")]

# -----------------------------------------------------------
# 2. MODELS (Pipeline: impute → scale → SelectKBest → model)
# -----------------------------------------------------------
def make_pipeline(model):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
        ("sel", SelectKBest(score_func=f_classif, k=20)),
        ("mdl", model)
    ])

baseline_models = {
    "Logistic Regression":        make_pipeline(LogisticRegression(max_iter=2000)),
    "Weighted Logistic Regression": make_pipeline(
                                    LogisticRegression(max_iter=2000, class_weight="balanced")),
    "Decision Tree":              make_pipeline(DecisionTreeClassifier(random_state=42)),
    "Random Forest":              make_pipeline(
                                    RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)),
    "Balanced Random Forest":     make_pipeline(
                                    RandomForestClassifier(n_estimators=400, random_state=42,
                                                          n_jobs=-1, class_weight="balanced"))
}

# -----------------------------------------------------------
# 3. PARAMETER GRIDS
# -----------------------------------------------------------
max_k = len(feature_cols)
param_grids = {
    "Logistic Regression": {
        "sel__k": randint(10, max_k+1),
        "mdl__C": uniform(0.01, 9.99),
        "mdl__penalty": ["l2","l1"],
        "mdl__solver":  ["lbfgs","liblinear"]
    },
    "Weighted Logistic Regression": {
        "sel__k": randint(10, max_k+1),
        "mdl__C": uniform(0.01, 9.99),
        "mdl__penalty": ["l2","l1"],
        "mdl__solver":  ["lbfgs","liblinear"]
    },
    "Decision Tree": {
        "sel__k": randint(10, max_k+1),
        "mdl__max_depth":         randint(2,12),
        "mdl__min_samples_split": randint(2,12),
        "mdl__min_samples_leaf":  randint(1,6),
        "mdl__criterion":         ["gini","entropy"]
    },
    "Random Forest": {
        "sel__k": randint(10, max_k+1),
        "mdl__n_estimators":      randint(150,601),
        "mdl__max_depth":         randint(3,18),
        "mdl__min_samples_split": randint(2,12),
        "mdl__min_samples_leaf":  randint(1,6),
        "mdl__max_features":      ["sqrt","log2",None]
    },
    "Balanced Random Forest": {
        "sel__k": randint(10, max_k+1),
        "mdl__n_estimators":      randint(150,601),
        "mdl__max_depth":         randint(3,18),
        "mdl__min_samples_split": randint(2,12),
        "mdl__min_samples_leaf":  randint(1,6),
        "mdl__max_features":      ["sqrt","log2",None]
    }
}

# -----------------------------------------------------------
# 4. BACK-TEST helper   (with optional tag for printing)
# -----------------------------------------------------------

def hit_table(model_dict, tag=""):
    acc = defaultdict(dict)
    for name, mdl in model_dict.items():
        for yr in range(2013, 2025):
            train = pd.concat([dfs_prepared[f"df_{y}"] for y in range(2012, yr)])
            test  = dfs_prepared[f"df_{yr}"]
            mdl.fit(train[feature_cols].fillna(0), train["MVP"])
            scores = (mdl.predict_proba(test[feature_cols].fillna(0))[:, 1]
                      if hasattr(mdl, "predict_proba")
                      else mdl.predict(test[feature_cols].fillna(0)))
            acc[name][yr] = int(test.loc[np.argmax(scores), "Player"] == mvp_dict[yr])

        # הדפסה אופציונלית
        if tag:
            hits = list(acc[name].values())
            print(f"{tag:<4} {name:<28}: {sum(hits)}/{len(hits)}  "
                  f"({100*np.mean(hits):.1f}%)")

    return acc

baseline_df = pd.DataFrame(hit_table(baseline_models)).T
baseline_df["hit_rate"] = baseline_df.mean(axis=1)
# -----------------------------------------------------------
# 5. Hyper-Tuning  (GroupKFold + hit-rate scorer)
# -----------------------------------------------------------
def hr_score(y_true, y_proba):
    return int(y_true.iloc[np.argmax(y_proba)] == 1)

hr_scorer = make_scorer(hr_score, needs_proba=True)

X_all = pd.concat([dfs_prepared[f"df_{y}"][feature_cols] for y in range(2012, 2025)],
                  ignore_index=True).fillna(0)
y_all = pd.concat([dfs_prepared[f"df_{y}"]["MVP"] for y in range(2012, 2025)],
                  ignore_index=True)

season_groups = pd.concat(
    [pd.Series([yr]*len(dfs_prepared[f"df_{yr}"])) for yr in range(2012, 2025)],
    ignore_index=True
)
gkf = GroupKFold(n_splits=min(6, season_groups.nunique()))

tuned_models, searches = {}, {}
for name, base in baseline_models.items():
    if not param_grids[name]:                   
        tuned_models[name], searches[name] = base, None
        continue
    rs = RandomizedSearchCV(
        base, param_grids[name], n_iter=25, scoring=hr_scorer,
        cv=gkf.split(X_all, y_all, groups=season_groups),
        n_jobs=-1, random_state=42
    )
    rs.fit(X_all, y_all)
    tuned_models[name], searches[name] = rs.best_estimator_, rs

tuned_df = pd.DataFrame(hit_table(tuned_models)).T
tuned_df["hit_rate"] = tuned_df.mean(axis=1)

# -----------------------------------------------------------
# 6. Selected features per tuned model
# -----------------------------------------------------------
print("\nSelected features per tuned model:")
for name, pipe in tuned_models.items():
    if "sel" in pipe.named_steps:
        mask = pipe.named_steps["sel"].get_support()
        feats = [f for f, keep in zip(feature_cols, mask) if keep]
        print(f"{name:<28} k={mask.sum():>2} | {feats}")
    else:
        print(f"{name:<28} (no selector)")

# -----------------------------------------------------------
# 7. Choose best model 
# -----------------------------------------------------------
compare = tuned_df["hit_rate"].sort_values(ascending=False)
best_name = next(m for m in compare.index if m != "Linear Regression")
best_model = tuned_models[best_name]

print(f"\nBest model: {best_name}  –  Hit-rate = {compare[best_name]:.3f}")

#  (the remainder of your script – predictions, plots, CSV saves, etc. – can follow here)

# ---------- 8. BACK-TEST (Tuned) ----------
print("\n==== BACK-TEST AFTER TUNING (2013-2024) ====")
tuned_acc = hit_table(tuned_models, tag="TUNE")
tuned_df  = pd.DataFrame(tuned_acc).T
tuned_df["hit_rate"] = tuned_df.mean(axis=1)

# ---------- 9. COMPARISON & PARAM-DIFF ----------
compare = pd.DataFrame({
    "Baseline_hit": baseline_df["hit_rate"],
    "Tuned_hit":    tuned_df["hit_rate"],
    "Gain":       tuned_df["hit_rate"] - baseline_df["hit_rate"]
}).sort_values("Tuned_hit", ascending=False)

param_rows = []
for n in baseline_models:
    bp,tp = {},{}
    if param_grids[n]:
        bp = {k: baseline_models[n].get_params().get(k) for k in param_grids[n]}
        tp = {k: tuned_models[n].get_params().get(k)    for k in param_grids[n]}
    param_rows.append({"Model": n,
                       "Baseline params": bp,
                       "Tuned params": tp,
                       "Gain": compare.loc[n,"Gain"]})
params_df = pd.DataFrame(param_rows)

print("\n=========== HIT-RATE COMPARISON ===========")
print(compare.round(3))
print("\n=========== PARAMETER CHANGES =============")
pd.set_option("display.max_colwidth", None)
print(params_df[["Model","Baseline params","Tuned params","Gain"]])

# ----------10. OVER-FITTING DIAGNOSTICS ----------
print("\n=== TRAIN vs CV Accuracy (gap>0 ⇒ חשד Over-fit) ===")
for n,m in tuned_models.items():
    cv = cross_validate(m, X_all, y_all,
                        cv=gkf.split(X_all,y_all,groups=season_groups),
                        scoring="accuracy", return_train_score=True, n_jobs=-1)
    print(f"{n:<28} train={cv['train_score'].mean():.3f} | "
          f"cv={cv['test_score'].mean():.3f} | "
          f"gap={cv['train_score'].mean()-cv['test_score'].mean():.3f}")

# ----------11. BEST MODEL & MVP 2025 ----------
best_name = compare.index[0]
best_model = tuned_models[best_name]
test_25 = dfs_prepared["df_2025"].copy()
scores_25 = (best_model.predict_proba(test_25[feature_cols].fillna(0).replace([np.inf,-np.inf],0))[:,1]
             if hasattr(best_model,"predict_proba")
             else best_model.predict(test_25[feature_cols].fillna(0).replace([np.inf,-np.inf],0)))
pred_2025 = test_25.loc[np.argmax(scores_25),"Player"]
print(f"\n🏆 BEST MODEL: {best_name}  ({100*compare.iloc[0]['Tuned_hit']:.1f}% hit-rate)")
print(f"🔮 Predicted MVP 2025: {pred_2025}")

# ----------12. TOP-10 PER YEAR (best model) ----------
top10_yearly = defaultdict(list)
for yr in range(2013, 2025):
    train = pd.concat([dfs_prepared[f"df_{y}"] for y in range(2012, yr)])
    best_model.fit(train[feature_cols].fillna(0).replace([np.inf,-np.inf],0),
                   train["MVP"])
    tst = dfs_prepared[f"df_{yr}"].copy()
    sc  = (best_model.predict_proba(tst[feature_cols].fillna(0).replace([np.inf,-np.inf],0))[:,1]
           if hasattr(best_model,"predict_proba")
           else best_model.predict(tst[feature_cols].fillna(0).replace([np.inf,-np.inf],0)))
    tst["Score"] = sc
    top10_yearly[yr] = tst.sort_values("Score", ascending=False)\
                          .head(10)["Player"].tolist()

# ----------13. SAVE ARTIFACTS ----------
compare.to_csv("mvp_model_comparison.csv")
params_df.to_csv("mvp_params_change.csv", index=False)
pathlib.Path("top10_per_year_best_tuned.json").write_text(
    json.dumps(top10_yearly, indent=2, ensure_ascii=False))
print("\n💾 Saved: mvp_model_comparison.csv, mvp_params_change.csv, top10_per_year_best_tuned.json")

# ===========================================================
#                 LINKEDIN-READY PLOTS
# ===========================================================
top_models = list(tuned_df["hit_rate"].sort_values(ascending=False).head(4).index)

# A. Hit-Rate Barplot
plt.figure(figsize=(7,5))
plt.bar(top_models, tuned_df.loc[top_models,"hit_rate"])
plt.ylabel("Hit-Rate (2013-2024)")
plt.title("🏀 Top 4 Models – Historical Hit-Rate")
plt.xticks(rotation=25); plt.tight_layout()
plt.savefig("plot_hit_rate.png", dpi=300)
plt.show()

# B. 2025 Prediction Barplot
mvp_probs, mvp_names = [], []
for m in top_models:
    mdl = tuned_models[m]
    scores = (mdl.predict_proba(test_25[feature_cols].fillna(0))[:,1]
              if hasattr(mdl,"predict_proba")
              else mdl.predict(test_25[feature_cols].fillna(0)))
    idx = np.argmax(scores)
    mvp_probs.append(scores[idx])
    mvp_names.append(test_25.loc[idx,"Player"])

plt.figure(figsize=(7,5))
plt.bar(top_models, mvp_probs)
for i,(p,name) in enumerate(zip(mvp_probs, mvp_names)):
    plt.text(i, p+0.01, name, ha='center', va='bottom', fontsize=8, rotation=90)
plt.ylabel("Predicted Probability / Score")
plt.title("🔮 2025 MVP Prediction – Top 4 Models")
plt.xticks(rotation=25); plt.tight_layout()
plt.savefig("plot_2025_prediction.png", dpi=300)
plt.show()

# C. Year-by-Year Hit Timeline
years = list(range(2013, 2025))
plt.figure(figsize=(9,5))
for m in top_models:
    hits = [tuned_acc[m][y] for y in years]
    plt.plot(years, hits, marker='o', label=m)
plt.yticks([0,1]); plt.xlabel("Season"); plt.ylabel("Hit (1 = Correct MVP)")
plt.title("📅 Year-by-Year MVP Hit – Top 4 Models")
plt.legend(); plt.tight_layout()
plt.savefig("plot_hit_timeline.png", dpi=300)
plt.show()

# ===========================================================
#   EXTRA: probabilities for EVERY player, year & model
# ===========================================================
print("\n📦  Building full probability table …")

player_prob_rows = []         

all_years = list(range(2013, 2026))

for mdl_name, mdl in tuned_models.items():
    for yr in all_years:
        df_year = dfs_prepared[f"df_{yr}"].copy()
        X_year = df_year[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        
        probs = (mdl.predict_proba(X_year)[:, 1] 
                 if hasattr(mdl, "predict_proba") 
                 else mdl.predict(X_year))
        
        # הוספת שורות לטבלה
        for player, prob in zip(df_year["Player"], probs):
            player_prob_rows.append({
                "Year": yr,
                "Model": mdl_name,
                "Player": player,
                "Probability": prob
            })

prob_df = pd.DataFrame(player_prob_rows)
csv_path = "player_mvp_probabilities.csv"
prob_df.to_csv(csv_path, index=False)
print(f"✅ Saved: {csv_path}  ({prob_df.shape[0]:,} rows)")

# ----------  EXTRA PRINT: TOP-10 ----------
print(f"\n🏅  TOP-10 per season – best model: {best_name}\n")

for yr in sorted(prob_df["Year"].unique()):
    # 1) סינון שנה + מודל
    df_y = prob_df[(prob_df["Year"] == yr) &
                   (prob_df["Model"] == best_name)]

    # 2) מיון לפי הסתברות
    top10 = df_y.sort_values("Probability", ascending=False).head(10)

    # 3) בניית מחרוזת “שם (0.42)”
    players_str = ", ".join(
        f"{row.Player} ({row.Probability:.2f})"
        for _, row in top10.iterrows()
    )

    # 4) הדפסה
    print(f"{yr}: {players_str}")


# ===========================================================
#heat map mvp candidates
# ===========================================================
import seaborn as sns
import matplotlib.pyplot as plt

top_players = [
    "LeBron James",
    "Kevin Durant",
    "Stephen Curry",
    "James Harden",
    "Russell Westbrook",
    "Kawhi Leonard",
    "Giannis Antetokounmpo",
    "Joel Embiid",
    "Nikola Jokić",
    "Luka Dončić",
    "Jayson Tatum",
    "Shai Gilgeous-Alexander",
    "Anthony Davis",
    "Anthony Edwards",
    "Damian Lillard",
    "Devin Booker",
    "Jimmy Butler",
    "Kyrie Irving",
    "Zion Williamson",
    "Cade Cunningham"
]

model_order = tuned_df.sort_values("hit_rate", ascending=False).index

for player_name in top_players:
    player_df = prob_df[prob_df["Player"] == player_name].copy()
    player_df = player_df.drop_duplicates(subset=["Model", "Year"], keep="first")
    pivot_df = player_df.pivot(index="Model", columns="Year", values="Probability")
    pivot_df = pivot_df.reindex(index=model_order).fillna(0)

    plt.figure(figsize=(12, 4))
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        linewidths=0.5,
        vmin=0,        
        vmax=1          
    )
    plt.title(f"{player_name} – MVP Probability by Model & Year")
    plt.xlabel("Season")
    plt.ylabel("Model")
    plt.tight_layout()

    safe_name = player_name.replace(" ", "_").replace("ć", "c").replace("č", "c").replace("í", "i")
    filename = f"heatmap_{safe_name.lower()}_mvp_probs.png"
    plt.savefig(filename, dpi=300)
    plt.show()
    

# ===========================================================
#bar plot
# ===========================================================
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.barplot(
    data=tuned_df.sort_values("hit_rate", ascending=False).reset_index(),
    x="hit_rate",
    y="index",
    palette="viridis"
)
plt.xlabel("Hit Rate (MVP Accuracy)")
plt.ylabel("Model")
plt.title("Comparison of MVP Prediction Accuracy (2013–2024)")
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig("barplot_mvp_hit_rate.png", dpi=300)
plt.show() 
    
    
    
    
#==============================
#boxplot
#=============================
df_2025 = prob_df[prob_df["Year"] == 2025].copy()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df_2025, x="Model", y="Probability", palette="Set3")
plt.xticks(rotation=45)
plt.ylabel("Predicted MVP Probability")
plt.title("Distribution of MVP Predictions for 2025 by Model")
plt.tight_layout()
plt.savefig("boxplot_2025_mvp_predictions.png", dpi=300)
plt.show()
    
    
    
    
# ===========================================================
#heat map
# ===========================================================
# Drop duplicates before pivoting
df_2025 = prob_df[prob_df["Year"] == 2025].copy()
df_2025 = df_2025.drop_duplicates(subset=["Player", "Model"], keep="first")

# Pivot: players as rows, models as columns
df_2025_pivot = df_2025.pivot(index="Player", columns="Model", values="Probability").fillna(0)

# Correlation between models
correlation_matrix = df_2025_pivot.corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Model Correlation – MVP Prediction Patterns (2025)")
plt.tight_layout()
plt.savefig("heatmap_model_correlation_2025.png", dpi=300)
plt.show()
    
    
    
    

    
