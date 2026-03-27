# Predicting Customer Churn for Lloyd's Banking Group

This project tackles the critical business problem of customer churn. By analyzing customer data, we've built a machine learning model that can predict which customers are likely to leave. This allows for proactive, targeted retention efforts, which can save the bank money and improve customer loyalty.

This README provides a high-level overview of our approach, findings, and how this work can be used to make a real-world impact.

## The Big Picture: Our Approach

Our goal was to build a reliable churn prediction model. We broke this down into two main phases:

1.  **Task 1: Data Foundation & Exploration.** We started by gathering and cleaning data from various sources, including customer demographics, transaction histories, and service interactions. We then explored this data to understand the underlying patterns and identify what factors might be related to churn. This foundational work was crucial for building a meaningful model.

2.  **Task 2: Building and Evaluating the Model.** With a clean dataset, we tested several machine learning algorithms to see which one could best predict churn. We selected the most promising model, fine-tuned it for optimal performance, and then dug into its predictions to understand *why* it was making them.

## What We Did and Why

### Data Preparation & Understanding (Task 1)

Before we could even think about prediction, we had to get our data in order. We pulled together information from five different datasets, covering everything from a customer's age and income to their transaction frequency and service complaint history.

We aggregated this data to create a single, comprehensive view of each customer. This involved calculating new features like `total_spend`, `complaint_count`, and `LoginFrequency`. We then carefully cleaned the data, handling missing values and outliers to ensure our model wouldn't be led astray by messy or incomplete information.

This exploratory phase gave us our first clues about churn. For instance, we saw that lower login frequency and certain types of service usage were correlated with a higher likelihood of churning.

### Predicting Churn (Task 2)

With our customer-level dataset ready, we moved on to the predictive modeling phase.

**Choosing the Right Tool for the Job:**

We evaluated four different machine learning models: Logistic Regression, Decision Tree, Gradient Boosting, and Random Forest. We chose **Random Forest** as our final model.

**Why Random Forest?** It consistently showed the best ability to distinguish between customers who would churn and those who wouldn't (as measured by ROC-AUC score). It's also a robust model that is less prone to overfitting and provides clear insights into which factors are most important for prediction.

**Fine-Tuning and Performance:**

We tuned the Random Forest model to get the best possible performance. On our holdout test set, the model's performance was modest, which highlights a key challenge: the available data only tells part of the story. While the model provides a valuable signal, its accuracy is limited by the predictive power of the features we have.

**The Key Drivers of Churn:**

The model told us which factors were the most important for predicting churn. The top drivers include:

1.  `avg_spend`: How much a customer typically spends.
2.  `LoginFrequency`: How often they engage with online services.
3.  `Age`: The customer's age.
4.  `svc_count`: The number of times they've contacted customer service.
5.  `unresolved_rate`: The proportion of their service issues that went unresolved.

These insights are valuable on their own, as they point to specific areas of the customer experience that are linked to loyalty.

## Where Do We Go From Here? (Future Scope)

This project is a strong starting point, but there's always room for improvement. Here are our key recommendations for future work:

*   **Richer Features:** The biggest opportunity for improvement is to create more powerful features. This could include:
    *   **Trend Features:** Are transactions increasing or decreasing over time?
    *   **Recency Features:** How many days has it been since a customer's last login or transaction?
    *   **Product Breadth:** How many different products or services does a customer use?

*   **Advanced Modeling:** We could explore more advanced algorithms like XGBoost or LightGBM, which often outperform Random Forest. We could also experiment with building separate models for different customer segments (e.g., a model just for mobile app users).

*   **Better Imbalance Handling:** Since churn is a relatively rare event, our dataset is imbalanced. Using techniques like SMOTE (Synthetic Minority Oversampling Technique) could help the model learn the patterns of the minority "churn" class more effectively.

## Limitations and What We Can Improve

It's important to be transparent about the limitations of this work:

*   **Limited Predictive Signal:** The current features, while useful, don't capture all the nuances of why a customer might leave. This is the primary reason for the model's modest performance on the test set. The path to a more accurate model is through better data and feature engineering.
*   **Small Dataset:** With only 1,000 customers, the model's performance metrics can be sensitive to the specific customers in the test set. A larger dataset would lead to more stable and reliable results.
*   **Static View:** The current model is based on a static snapshot of customer data. A real-world implementation should use a more dynamic, time-based approach to validation (e.g., training on past data to predict future churn).

By investing in richer data and exploring more advanced techniques, we can build on this foundation to create an even more powerful and impactful churn prediction tool.
# LLoyd-Banking-Churn-Project
