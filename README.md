# Predictive Modeling for NBA MVP Selection (2012–2025)

## Abstract
This repository presents a sophisticated predictive framework for identifying the National Basketball Association (NBA) Most Valuable Player (MVP) from regular-season player performance data spanning 2012 to 2025. Leveraging advanced statistical standardization, machine learning algorithms, and a custom evaluation metric, the model achieves robust predictive performance, with a balanced random forest algorithm accurately identifying the MVP in approximately 58% of tested seasons. The 2025 analysis strongly favors Nikola Jokić as the leading candidate, supported by multiple models. This work demonstrates a scalable, data-driven approach to ranking elite performance in professional basketball.

## Introduction
The NBA MVP award represents the pinnacle of individual achievement in professional basketball, determined annually by a vote among media members. This project aims to replicate and predict this selection process through a data-driven methodology, utilizing comprehensive player performance data from Basketball Reference (2012–2025). By standardizing statistics across seasons, addressing class imbalance, and deploying a suite of machine learning models, this framework offers a rigorous, reproducible approach to MVP prediction. The repository encapsulates the data processing pipeline, model implementations, and predictive outcomes, with an emphasis on methodological transparency and extensibility.

## Methodology
The methodology integrates data preprocessing, feature engineering, model selection, and evaluation strategies tailored to the challenges of MVP prediction.

### Data Acquisition and Preprocessing
- **Source**: Regular-season box score statistics for all NBA players from 2012 to 2025 were retrieved from Basketball Reference.
- **Standardization**: To account for variations in game pace and statistical norms across seasons, each performance metric (e.g., points, rebounds) was z-scored relative to the league average for its respective season.
- **Labeling**: A binary target variable was assigned, with `1` indicating the actual MVP for a given season and `0` for all other players, resulting in a highly imbalanced dataset (approximately 0.3% positive cases).

### Model Selection
Five machine learning algorithms were employed to address the prediction task:
1. Logistic Regression (LR)
2. Class-Weighted Logistic Regression (WLR)
3. Decision Tree (DT)
4. Random Forest (RF)
5. Balanced Random Forest (BRF)

### Evaluation Strategy
- **Custom Metric**: To mitigate the impact of class imbalance, a bespoke scoring function was designed to evaluate models based solely on whether the top-ranked player matched the actual MVP for a given season.
- **Cross-Validation**: A "leave-one-season-out" scheme was implemented, training models on 11 consecutive seasons to predict the 12th, iterated across all seasons (12 folds total).
- **Performance Metric**: The hit rate, defined as the proportion of seasons where the predicted MVP matched the actual MVP, was used to compare model performance.

## Procedure
The analytical pipeline was executed as follows:
1. **Data Preparation**: Raw box score data were cleaned, aggregated, and z-scored to ensure comparability across seasons.
2. **Feature Engineering**: Standardized performance metrics were used as input features, with the binary MVP label as the target.
3. **Model Training and Evaluation**: Each algorithm was trained and evaluated using the leave-one-season-out cross-validation scheme. Hyperparameters were tuned via grid search to optimize the custom scorer.
4. **2025 Prediction**: The trained models were applied to the 2025 season data to generate MVP probabilities for all eligible players.
5. **Visualization and Interpretation**: Predictive outputs were analyzed to identify top candidates and assess model agreement.

## Results
The balanced random forest model achieved the highest performance, correctly identifying the MVP in 7 out of 12 seasons (hit rate ≈ 58%). The weighted logistic regression and standard random forest models followed closely, each with 6 correct predictions. The decision tree exhibited lower performance but provided valuable diversity for potential ensemble methods. For the 2025 season, four of the five models ranked Nikola Jokić as the top candidate, with a median win probability of approximately 68%. Giannis Antetokounmpo and Shai Gilgeous-Alexander emerged as secondary contenders, though with significantly lower probabilities.

## Conclusions
This project demonstrates a robust, data-driven approach to predicting the NBA MVP, leveraging standardized performance metrics and advanced machine learning techniques. The balanced random forest model's superior performance underscores the efficacy of ensemble methods in handling imbalanced datasets. The strong prediction of Nikola Jokić as the 2025 MVP reflects the framework's ability to identify elite performers based on historical and current data. Future enhancements include:
- Integration of advanced metrics (e.g., RAPTOR, EPM).
- Exploration of alternative algorithms, such as focal loss XGBoost and LightGBM.
- Development of a Streamlit dashboard for real-time MVP probability updates.

This repository provides a foundation for further research and application in sports analytics, with potential extensions to other performance-based ranking tasks.