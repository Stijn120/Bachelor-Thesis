import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sci
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import MultiComparison

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)

survey_data = pd.read_csv("Bachelor Thesis Phosphene Head Pose Detection_December 6, 2020_02.57.csv")
columns = survey_data.columns

email_adresses = survey_data['Email'][2:]

#Remove unneccessary columns
columns_to_drop = []

firstClick = re.compile('.+TimerExp_First Click$')
answer = re.compile('.+ResSecExp$')
condition = re.compile('.+Condition$')
correctAns = re.compile('.+CorrectAns$')

for column in columns:

    if not column == "ParticipantName" and not column == "Duration" and not column == 'Gender' and not column == 'Age' and not column == 'Consent' and not bool(re.match(firstClick, column)) and not bool(re.match(answer, column)) and not bool(re.match(condition, column)) and not bool(re.match(correctAns, column)):
        columns_to_drop.append(column)

simple_survey_data = survey_data.drop(columns=columns_to_drop)

def exportConsent():
    names = survey_data['ParticipantName'][2:]
    birthdays = survey_data['ParticipantBirthday'][2:]
    timestamps = survey_data['RecordedDate'][2:]
    consent = survey_data['Consent'][2:]

    consentInformation = pd.DataFrame(list(zip(names, birthdays, timestamps, consent)), columns=['Name', 'Birthday', 'Timestamp', 'Consent'])
    consentInformation.to_csv('ConsentInformation.csv')


def answerEqualsLabel(response, label):

    if response == "Front" and label == str(0):
        return True
    elif response == "Left" and label == str(1):
        return True
    elif response == "Right" and label == str(2):
        return True
    elif response == "Down" and label == str(3):
        return True
    elif response == "Up" and label == str(4):
        return True
    elif response == "Down Right" and label == str(5):
        return True
    elif response == "Down Left" and label == str(6):
        return True
    elif response == "Up Right" and label == str(7):
        return True
    elif response == "Up Left" and label == str(8):
        return True
    else:
        return False

def translateLabelsToDirection(labels):
    directions = []

    for label in labels:
        if label == str(0):
            directions.append("Front")
        elif label == str(1):
            directions.append("Left")
        elif label == str(2):
            directions.append("Right")
        elif label == str(3):
            directions.append("Down")
        elif label == str(4):
            directions.append("Up")
        elif label == str(5):
            directions.append("Down Right")
        elif label == str(6):
            directions.append("Down Left")
        elif label == str(7):
            directions.append("Up Right")
        elif label == str(8):
            directions.append("Up Left")

    return directions

def calcMeasures(data, cond, specificParticipant):
    columns = data.columns
    nrOfColumns = len(columns)

    accuracies = []
    reactionTimes = []
    accuraciesList = []
    reactionTimesList = []
    misclassified = []

    for participant in range(2, len(data.index)):

        participantCorrectResponses = []
        participantReactionTimes = []
        if specificParticipant > -1:
            participant = specificParticipant+1

        #print(data)

        for columnNr in range(nrOfColumns):

            if bool(re.match(condition, columns[columnNr])) and data[columns[columnNr]][participant] == str(cond):         #Get condition columnNr

                if answerEqualsLabel(data.iloc[:,columnNr-3][participant], data.iloc[:,columnNr-1][participant]):
                    participantCorrectResponses.append(1)
                    participantReactionTimes.append(float(data.iloc[:,columnNr-2][participant]))
                else:
                    participantCorrectResponses.append(0)
                    participantReactionTimes.append(float(data.iloc[:, columnNr - 2][participant]))
                    misclassified.append(data.iloc[:,columnNr-1][participant])

        participantAvgReactionTime = np.average(participantReactionTimes)
        reactionTimes.append(participantAvgReactionTime)
        reactionTimesList.append(participantReactionTimes)
        participantAccuracy = np.average(participantCorrectResponses)
        accuracies.append(participantAccuracy)
        accuraciesList.append(participantCorrectResponses)

    averageReactionTime = np.average(reactionTimes)
    averageAccuracy = np.average(accuracies)
    print("Average Accuracy of Condition {0} is: {1}".format(cond, round(averageAccuracy, 3)))
    print("Average Reaction Time of Condition {0} is: {1}".format(cond, round(averageReactionTime, 3)))
    print("")

    return averageAccuracy, averageReactionTime, accuracies, reactionTimes, accuraciesList, reactionTimesList, misclassified


def plotData(results, results2, misclassificationsdf):
    # Participant Age Distribution
    sns.histplot(results['age'].sort_values())
    plt.title("Participant Age Distribution")
    plt.xlabel("Age")
    plt.savefig('AgeDist.png', format='png')
    plt.show()

    # Participant Gender Distribution
    male_count = results['gender'].tolist().count('Male')
    female_count = results['gender'].tolist().count('Female')
    sns.barplot(x=['male', 'female'], y=[male_count, female_count])
    plt.title("Participant Gender Distribution")
    plt.savefig('GenderDist.png', format='png')
    plt.show()

    # Reaction Times for Different Genders
    sns.distplot(results['r0'][results['gender'] == 'Male'], label='Male Edge Detection', hist=False)
    sns.distplot(results['r0'][results['gender'] == 'Female'], label='Female Edge Detection', hist=False)
    sns.distplot(results['r1'][results['gender'] == 'Male'], label='Male Complexity 1', hist=False)
    sns.distplot(results['r1'][results['gender'] == 'Female'], label='Female Complexity 1', hist=False)
    sns.distplot(results['r2'][results['gender'] == 'Male'], label='Male Complexity 2', hist=False)
    sns.distplot(results['r2'][results['gender'] == 'Female'], label='Female Complexity 2', hist=False)
    sns.distplot(results['r3'][results['gender'] == 'Male'], label='Male Complexity 3', hist=False)
    sns.distplot(results['r3'][results['gender'] == 'Female'], label='Female Complexity 3', hist=False)
    plt.legend()
    plt.title("Reaction Times for Different Genders")
    plt.savefig('RTGender.png', format='png')
    plt.show()

    # Accuracies for Different Genders
    sns.distplot(results['a0'][results['gender'] == 'Male'], label='Male Edge Detection', hist=False)
    sns.distplot(results['a0'][results['gender'] == 'Female'], label='Female Edge Detection', hist=False)
    sns.distplot(results['a1'][results['gender'] == 'Male'], label='Male Complexity 1', hist=False)
    sns.distplot(results['a1'][results['gender'] == 'Female'], label='Female Complexity 1', hist=False)
    sns.distplot(results['a2'][results['gender'] == 'Male'], label='Male Complexity 2', hist=False)
    sns.distplot(results['a2'][results['gender'] == 'Female'], label='Female Complexity 2', hist=False)
    sns.distplot(results['a3'][results['gender'] == 'Male'], label='Male Complexity 3', hist=False)
    sns.distplot(results['a3'][results['gender'] == 'Female'], label='Female Complexity 3', hist=False)
    plt.legend()
    plt.title("Accuracies for Different Genders")
    plt.savefig('ACCGender.png', format='png')
    plt.show()

    # Reaction Times for Different Ages
    plt.scatter(results['age'].sort_values(), results['r0'], label='Edge Detection')
    plt.scatter(results['age'].sort_values(), results['r1'], label='Vertex Processing Complexity 1')
    plt.scatter(results['age'].sort_values(), results['r2'], label='Vertex Processing Complexity 2')
    plt.scatter(results['age'].sort_values(), results['r3'], label='Vertex Processing Complexity 3')
    plt.legend()
    plt.title("Reaction Times for Different Ages")
    plt.savefig('RTAge.png', format='png')
    plt.show()

    # Accuracies for Different Ages
    plt.scatter(results['age'].sort_values(), results['a0'], label='Edge Detection')
    plt.scatter(results['age'].sort_values(), results['a1'], label='Vertex Processing Complexity 1')
    plt.scatter(results['age'].sort_values(), results['a2'], label='Vertex Processing Complexity 2')
    plt.scatter(results['age'].sort_values(), results['a3'], label='Vertex Processing Complexity 3')
    plt.legend()
    plt.title("Accuracies for Different Ages")
    plt.savefig('ACCAge.png', format='png')
    plt.show()

    # Accuracy Boxplot
    plt.figure(figsize=[7, 6])
    sns.boxplot(x='Condition', y='Accuracy', data=results2)
    plt.xlabel('Condition')
    plt.ylabel('Accuracy')
    plt.suptitle("Accuracy Boxplot")
    plt.title("Anova F(3, 144)=150.33, p<0.0001")

    x1, x2 = 0, 1
    y, h, col = results2['Accuracy'].max() + 0.02, 0.01, 'k'
    plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    plt.text((x1 + x2) * .5, y + h, '****', ha='center', va='bottom', color=col)

    x1, x2 = 0, 2
    y, h, col = results2['Accuracy'].max() + 0.08, 0.01, 'k'
    plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    plt.text((x1 + x2) * .5, y + h, '****', ha='center', va='bottom', color=col)

    x1, x2 = 0, 3
    y, h, col = results2['Accuracy'].max() + 0.16, 0.01, 'k'
    plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    plt.text((x1 + x2) * .5, y + h, '****', ha='center', va='bottom', color=col)

    x1, x2 = 1, 2
    y, h, col = results2['Accuracy'].max() + 0.04, 0.01, 'k'
    plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    plt.text((x1 + x2) * .5, y + h, '***', ha='center', va='bottom', color=col)

    x1, x2 = 1, 3
    y, h, col = results2['Accuracy'].max() + 0.12, 0.01, 'k'
    plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    plt.text((x1 + x2) * .5, y + h, '****', ha='center', va='bottom', color=col)

    x1, x2 = 2, 3
    y, h, col = results2['Accuracy'].max() + 0.06, 0.01, 'k'
    plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    plt.text((x1 + x2) * .5, y + h, 'ns', ha='center', va='bottom', color=col)

    plt.savefig('ACCBoxplot.png', format='png')
    plt.show()

    # Reation Boxplot
    plt.figure(figsize=[7, 6])
    sns.boxplot(x='Condition', y='Reaction Time', data=results2)
    plt.xlabel('Condition')
    plt.ylabel('Reaction Time')
    plt.suptitle("Reaction Time Boxplot")
    plt.title("Anova F(3, 144)=68.19, p<0.0001")

    x1, x2 = 0, 1
    y, h, col = results2['Reaction Time'].max() + 0.5, 0.05, 'k'
    plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    plt.text((x1 + x2) * .5, y + h, '****', ha='center', va='bottom', color=col)

    x1, x2 = 0, 2
    y, h, col = results2['Reaction Time'].max() + 1.0, 0.05, 'k'
    plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    plt.text((x1 + x2) * .5, y + h, '****', ha='center', va='bottom', color=col)

    x1, x2 = 0, 3
    y, h, col = results2['Reaction Time'].max() + 1.2, 0.05, 'k'
    plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    plt.text((x1 + x2) * .5, y + h, '****', ha='center', va='bottom', color=col)

    x1, x2 = 1, 2
    y, h, col = results2['Reaction Time'].max() + 0.3, 0.05, 'k'
    plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    plt.text((x1 + x2) * .5, y + h, '****', ha='center', va='bottom', color=col)

    x1, x2 = 1, 3
    y, h, col = results2['Reaction Time'].max() + 0.7, 0.05, 'k'
    plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    plt.text((x1 + x2) * .5, y + h, '****', ha='center', va='bottom', color=col)

    x1, x2 = 2, 3
    y, h, col = results2['Reaction Time'].max() + 0.1, 0.05, 'k'
    plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    plt.text((x1 + x2) * .5, y + h, '**', ha='center', va='bottom', color=col)

    plt.savefig('RTBoxplot.png', format='png')
    plt.show()

    # Accuracy Distributions
    sns.distplot(results['a0'], label='Edge Detection', hist=False)
    sns.distplot(results['a1'], label='Vertex Processing Complexity 1', hist=False)
    sns.distplot(results['a2'], label='Vertex Processing Complexity 2', hist=False)
    sns.distplot(results['a3'], label='Vertex Processing Complexity 3', hist=False)
    plt.xlabel('Accuracy')
    plt.ylabel('Count')
    plt.title("Accuracy Distribution")
    plt.legend()
    plt.savefig('ACCDist.png', format='png')
    plt.show()

    # Reation Time Distributions
    sns.distplot(results['r0'], label='Edge Detection', hist=False)
    sns.distplot(results['r1'], label='Vertex Processing Complexity 1', hist=False)
    sns.distplot(results['r2'], label='Vertex Processing Complexity 2', hist=False)
    sns.distplot(results['r3'], label='Vertex Processing Complexity 3', hist=False)
    plt.xlabel('Reaction Time (seconds)')
    plt.ylabel('Count')
    plt.title("Reaction Time Distribution")
    plt.legend()
    plt.savefig('RTDist.png', format='png')
    plt.show()

    # Plot Misclassifications
    sns.countplot(x="Direction", hue="Condition", data=misclassificationsdf)
    plt.title("Misclassifications")
    plt.tight_layout()
    plt.ylabel("Number of Misclassifications")
    plt.legend()
    plt.savefig('Misclassfications.png', format='png')
    plt.show()


def makeResultsDataFrame(data):
    averageAccuracy_0, averageReactionTime_0, accuracies_0, reactionTimes_0, accuraciesList_0, reactionTimesList_0, misclassified_0 = calcMeasures(data, 0, -1)
    averageAccuracy_1, averageReactionTime_1, accuracies_1, reactionTimes_1, accuraciesList_1, reactionTimesList_1, misclassified_1 = calcMeasures(data, 1, -1)
    averageAccuracy_2, averageReactionTime_2, accuracies_2, reactionTimes_2, accuraciesList_2, reactionTimesList_2, misclassified_2 = calcMeasures(data, 2, -1)
    averageAccuracy_3, averageReactionTime_3, accuracies_3, reactionTimes_3, accuraciesList_3, reactionTimesList_3, misclassified_3 = calcMeasures(data, 3, -1)

    gender = data['Gender'][2:]
    age = data['Age'][2:]

    misclassified_0 = translateLabelsToDirection(misclassified_0)
    misclassified_1 = translateLabelsToDirection(misclassified_1)
    misclassified_2 = translateLabelsToDirection(misclassified_2)
    misclassified_3 = translateLabelsToDirection(misclassified_3)

    misclassified = np.append(np.append(np.append(misclassified_0, misclassified_1), misclassified_2), misclassified_3)
    len0 = np.full(shape=len(misclassified_0), fill_value='Edge Detection')
    len1 = np.full(shape=len(misclassified_1), fill_value='Vertex Processing \n Complexity 1')
    len2 = np.full(shape=len(misclassified_2), fill_value='Vertex Processing \n Complexity 2')
    len3 = np.full(shape=len(misclassified_3), fill_value='Vertex Processing \n Complexity 3')
    condition = np.append(np.append(np.append(len0, len1), len2), len3)

    misclassificationsdf = pd.DataFrame(list(zip(misclassified, condition)), columns=['Direction', 'Condition'])

    # Plot Reaction Times over Time
    averagedReactionTimes_0 = np.average(reactionTimesList_0, axis=1)
    averagedReactionTimes_1 = np.average(reactionTimesList_1, axis=1)
    averagedReactionTimes_2 = np.average(reactionTimesList_2, axis=1)
    averagedReactionTimes_3 = np.average(reactionTimesList_3, axis=1)
    plt.scatter(np.arange(len(averagedReactionTimes_0)), averagedReactionTimes_0, label="Edge Detection")
    plt.scatter(np.arange(len(averagedReactionTimes_1)), averagedReactionTimes_1, label="Vertex Processing \n Complexity 1")
    plt.scatter(np.arange(len(averagedReactionTimes_2)), averagedReactionTimes_2, label="Vertex Processing \n Complexity 2")
    plt.scatter(np.arange(len(averagedReactionTimes_3)), averagedReactionTimes_3, label="Vertex Processing \n Complexity 3")
    plt.legend()
    plt.show()

    # Plot Accuracies over Time
    averagedAccuracies_0 = np.average(accuraciesList_0, axis=1)
    averagedAccuracies_1 = np.average(accuraciesList_1, axis=1)
    averagedAccuracies_2 = np.average(accuraciesList_2, axis=1)
    averagedAccuracies_3 = np.average(accuraciesList_3, axis=1)
    plt.scatter(np.arange(len(averagedAccuracies_0)), averagedAccuracies_0, label="Edge Detection")
    plt.scatter(np.arange(len(averagedAccuracies_1)), averagedAccuracies_1, label="Vertex Processing \n Complexity 1")
    plt.scatter(np.arange(len(averagedAccuracies_2)), averagedAccuracies_2, label="Vertex Processing \n Complexity 2")
    plt.scatter(np.arange(len(averagedAccuracies_3)), averagedAccuracies_3, label="Vertex Processing \n Complexity 3")
    plt.legend()
    plt.show()

    results_list = list(
        zip(gender, age, accuracies_0, accuracies_1, accuracies_2, accuracies_3, reactionTimes_0, reactionTimes_1, reactionTimes_2,
            reactionTimes_3))

    accuraciesAll = np.append(np.append(np.append(accuracies_0, accuracies_1), accuracies_2), accuracies_3)
    reactiontimesAll = np.append(np.append(np.append(reactionTimes_0, reactionTimes_1), reactionTimes_2), reactionTimes_3)

    len0 = np.full(shape=len(accuracies_0), fill_value='Edge Detection')
    len1 = np.full(shape=len(accuracies_1), fill_value='Vertex Processing \n Complexity 1')
    len2 = np.full(shape=len(accuracies_2), fill_value='Vertex Processing \n Complexity 2')
    len3 = np.full(shape=len(accuracies_3), fill_value='Vertex Processing \n Complexity 3')
    condition = np.append(np.append(np.append(len0, len1), len2), len3)

    subjects = np.arange(1, 50)
    subject_list = np.append(np.append(np.append(subjects, subjects), subjects), subjects)
    results2_list = list(zip(accuraciesAll, reactiontimesAll, subject_list, condition))

    results = pd.DataFrame(results_list, columns=['gender', 'age', 'a0', 'a1', 'a2', 'a3', 'r0', 'r1', 'r2', 'r3'])
    results2 = pd.DataFrame(results2_list, columns=['Accuracy', 'Reaction Time', 'Subject', 'Condition'])

    return results, results2, misclassificationsdf


def analyzeData(results2):

    print('Accuracy')
    print(AnovaRM(data=results2, depvar='Accuracy', subject='Subject', within=['Condition'], aggregate_func='mean').fit())

    MultiComp = MultiComparison(results2['Accuracy'], results2['Condition'])
    comp = MultiComp.allpairtest(sci.ttest_rel, method='bonf')
    print(comp[0])

    print('Reaction Time')
    print(AnovaRM(data=results2, depvar='Reaction Time', subject='Subject', within=['Condition'], aggregate_func='mean').fit())

    MultiComp = MultiComparison(results2['Reaction Time'], results2['Condition'])
    comp = MultiComp.allpairtest(sci.ttest_rel, method='bonf')
    print(comp[0])


#exportConsent()

results, results2, misclassificationsdf = makeResultsDataFrame(simple_survey_data)

analyzeData(results2)

plotData(results, results2, misclassificationsdf)






