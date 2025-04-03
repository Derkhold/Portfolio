# 💼 Quantitative Trading Portfolio

Bienvenue sur mon portfolio de projets en **finance de marché**, **trading algorithmique** et **machine learning appliqué aux données financières**.

Chaque projet a été conçu pour combiner **rigueur quantitative**, **codage propre en Python** et **utilisabilité réelle** dans des environnements de backtest ou de simulation live.  
Les frameworks utilisés incluent : `Backtrader`, `TA-Lib`, `Alpaca API`, `Keras`, `Streamlit`, `FPDF`, et bien d'autres.

---

## 🧠 Objectifs du Portfolio

- 🔍 Appliquer des méthodes de **machine learning** et d’analyse technique à des données de marché réelles
- 🧱 Développer une architecture de trading **modulaire, réutilisable et scalable**
- 📈 Tester des stratégies robustes via **des backtests fiables** et des indicateurs de performance détaillés
- 📊 Construire des outils de **visualisation interactive** pour le suivi des risques et des performances

---

## 📂 Projets Inclus

### 1. 🧠 Modular Trading Architecture
> Une architecture complète de trading algorithmique, construite en Python avec Backtrader

📁 [`/trading_architecture_project`](./trading_architecture_project)

- Architecture modulaire : `OrderManager`, `RiskManager`, `PositionManager`, `TradeMonitor`
- Intégration API de données Alpaca + cache local
- Backtests avec génération automatique de rapports PDF, CSV, JSON
- Stratégies personnalisées (Trend Following, Reversal, Momentum)

---

### 2. 📈 LSTM Forecasting on Forex
> Modèle LSTM pour la prédiction de la tendance EUR/USD

📁 [`/forex_lstm_prediction`](./forex_lstm_prediction)

- Données historiques Forex nettoyées et normalisées
- Séquences temporelles pour apprentissage supervisé
- Architecture LSTM avec Keras/Tensorflow
- Évaluation des performances : RMSE, MAE, visualisation des erreurs

---

### 3. 📰 Crypto Sentiment Strategy
> Stratégie de trading basée sur l’analyse de sentiment Twitter dans les cryptomonnaies

📁 [`/crypto_sentiment_strategy`](./crypto_sentiment_strategy)

- Scraping de tweets et nettoyage NLP (TextBlob / Vader)
- Attribution de scores de sentiment et agrégation par période
- Génération de signaux d'achat/vente basés sur l'émotion du marché
- Backtest de stratégie sentimentale + rapport PDF final

---

### 4. 📊 Risk Metrics Dashboard
> Dashboard interactif avec calculs de risque personnalisés (VaR, CVaR, drawdown, etc.)

📁 [`/risk_metrics_dashboard`](./risk_metrics_dashboard)

- Application Streamlit avec calculs dynamiques
- Upload de données personnalisées ou sélection d’actifs simulés
- Visualisation de métriques de risque sur période glissante
- Export CSV + support pour analyse hors ligne

---

## 🧠 Technologies utilisées

- **Langage** : `Python 3.11`
- **Backtesting** : `Backtrader`, `Pandas`, `Matplotlib`
- **Machine Learning** : `Keras`, `Tensorflow`, `Scikit-learn`
- **API de données** : `Alpaca`, `Yahoo Finance`, `Twitter`
- **Analyse technique** : `TA-Lib`, indicateurs custom
- **Dashboards & Reporting** : `Streamlit`, `FPDF`, `JSON`, `CSV`

---

## 👨‍💻 À propos de moi

**Iséo Lasfargues**  
🎓 Étudiant en Master Finance de Marché – NEOMA Business School  
💡 Passionné par le quant trading, le machine learning, l’analyse de risque, et l’automatisation des stratégies  
📬 iseo.lasfargues.22@neoma-bs.com  
🔗 [LinkedIn](https://linkedin.com/in/tonprofil)  

---

## 📌 À venir

- [ ] Intégration d'un modèle transformer pour prédiction de tendance
- [ ] Déploiement live avec Alpaca Paper Trading
- [ ] Ajout de stratégies multi-actifs et multi-timeframes
- [ ] Comparaison de performance via GridSearch sur hyperparamètres

---

🧠 *Ce portfolio est en évolution constante. Chaque projet est pensé pour être **transparent, réutilisable et améliorable***.
