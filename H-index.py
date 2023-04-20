import pandas as pd
import numpy as np

def readData(x):
    if x ==0:
        pathGames = 'Data/MasseyGameData - MasseyExampleData.csv'
        pathTeams ='Data/MasseyGameData - teamsExample.csv'

    if x == 1:
        pathGames = 'Data/MasseyGameData - ProNational.csv'
        pathTeams = 'Data/MasseyGameData - NationalCProTeams.csv'

    if x == 2:
        pathGames = 'Data/MasseyGameData - AmericanConferenceProGames.csv'
        pathTeams = 'Data/MasseyGameData - AmericanCProTeams.csv'

    if x==3:
        pathGames ='Data/MasseyGameData - OnlyPro.csv'
        pathTeams= 'Data/MasseyGameData - ProTeams.csv'

    # game data
    # the winning team is in the TeamA column, in cases of a tie location is irrelevent
    data = pd.read_csv(pathGames)
    # list of teams
    teams = pd.read_csv(pathTeams)
    print("Game Data:\n",data)
    #print("\n",teams, "\n")

    return data, teams

def plotNormalCDF(x, arr):
    from scipy.stats import norm
    mean = np.mean(arr)
    std = pow((50* np.sum(arr)),0.25)
    dist = norm(mean, std)
    cdf = dist.cdf(x)
    return cdf

def calculateH(arr):
    n = len(arr)
    arr.sort(reverse=True)
    h_index = 0
    for i in range(n):
        if arr[i]>= i+1:
            h_index = i+1
        else:
            break
    return h_index



def makescoretable():
    s = {
        "team": teams.Team,
        "scores": [[] for x in range(len(teams))]
    }

    # load data into a DataFrame object:
    scores = pd.DataFrame(s)
    for g in data.index:
        A = data['TeamA'][g]
        B = data['TeamB'][g]
        indexA = np.where(teams.Team == A)[0][0]
        indexB = np.where(teams.Team == B)[0][0]
        cdf = plotNormalCDF(data['PointsA'][g], [data['PointsA'][g], data['PointsB'][g]])
        scores['scores'][indexA].append(cdf*6)
        scores['scores'][indexB].append((1-cdf)*6)
    print(scores)
    return scores




if __name__ == '__main__':
    dataset =3
    data, teams = readData(dataset)
    scores = makescoretable()
    scores['h_index'] = scores['scores'].apply(calculateH)
    print(scores.sort_values('h_index', ascending=False))