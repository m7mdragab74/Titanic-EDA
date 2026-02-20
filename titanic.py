import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Titanic-Dataset.csv')

# Drop Cabin (missing values)
df = df.drop(columns=['Cabin'])

# Fill missing values with mean
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
print("\nMissing Values:")
print(df.isnull().sum())

#survival rate
print("Survival Rate:", round(df['Survived'].mean(), 3))

#Survival by Sex
print("\nSurvival by Sex:")
print(df.groupby('Sex')['Survived'].mean().round(3))
print("\nInsight: Females have significantly higher survival rate.")

#Survival by Class
print("\nSurvival by Pclass:")
print(df.groupby('Pclass')['Survived'].mean())

#Survival by Age
sns.histplot(data=df, x='Age', hue='Survived', bins=20, kde=True)
plt.title("Survival by Age")
plt.show()

#Fare Distribution
plt.hist(df['Fare'], bins=20)
plt.title("Fare Distribution")
plt.show()

#Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

#Survival by Embarked
sns.barplot(x='Embarked', y='Survived', data=df)
plt.title("Survival Rate by Embarked Port")
plt.show()

#Survival by SibSp / Parch
sns.barplot(x='SibSp', y='Survived', data=df)
plt.title("Survival Rate by SibSp")
plt.show()

sns.barplot(x='Parch', y='Survived', data=df)
plt.title("Survival Rate by Parch")
plt.show()

#Visualization
df.groupby('Sex')['Survived'].mean().plot(kind='bar')
plt.title('Survival Rate by Sex')
plt.ylabel('Survival Rate')
plt.show()

sns.barplot(x='Pclass', y='Survived', data=df)
plt.title("Survival Rate by Passenger Class")
plt.show()