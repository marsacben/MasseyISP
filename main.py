import pandas as pd
import numpy as np
from tabulate import tabulate
import scipy.stats as stats

def finDir():
    import os
    for f in os.listdir("/"):
        print(f)

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
    print("\n",teams, "\n")

    return data, teams
def calcOffDef2(gof):
    # instantiating matrices
    y = np.zeros(shape=(len(data.PointsA)*2))
    X = np.zeros(shape=(len(data.PointsA)*2, len(teams)*2))
    # print(X, y)

    # setting matrix values based on data
    i=0
    for r in data.index:
        A = data['TeamA'][r]
        B = data['TeamB'][r]
        indexA= np.where(teams.Team == A)[0][0]
        indexB = np.where(teams.Team == B)[0][0]
        #points scored by A
        X[i][indexA* 2] = 1
        X[i][(indexB* 2) + 1] = -1
        #print(teams['PointsAgainst'][indexA], A, "hi", data['PointsA'][r], np.std(teams['PointsAgainst'][indexA]))
        pA = data['PointsA'][r]
        pB = data['PointsB'][r]
        if gof==0:
            y[i] = 1
        else:
            y[i] = GameOutcomeFunction(pA, [pA, pB])
        # points scored by B
        i +=1
        X[i][(indexA * 2) +1] = -1
        X[i][indexB* 2] = 1
        if gof==0:
            y[i] = -1
        else:
            y[i] = GameOutcomeFunction(pB, [pA, pB])
        i += 1

    return X, y

def calcOffDef():
    # instantiating matrices
    y = np.zeros(shape=(len(data.PointsA)*2))
    X = np.zeros(shape=(len(data.PointsA)*2, len(teams)*2))
    # print(X, y)

    # setting matrix values based on data
    i=0
    for r in data.index:
        A = data['TeamA'][r]
        B = data['TeamB'][r]
        indexA= np.where(teams.Team == A)[0][0] * 2
        indexB = np.where(teams.Team == B)[0][0] * 2
        #points scored by A
        X[i][indexA] = 1
        X[i][indexB + 1] = -1
        y[i]= data['PointsA'][r]
        # points scored by B
        i +=1
        X[i][indexA +1] = -1
        X[i][indexB] = 1
        y[i] = data['PointsB'][r]
        i += 1

    return X, y

def simpleLeastSquares():
    # instantiating matrices
    y = np.zeros(shape=(len(data.PointsA)))
    X = np.zeros(shape=(len(data.PointsA), len(teams)))
    # print(X, y)

    # setting matrix values based on data

    # for y
    for i in range(len(data.PointsA)):
        y[i] = data.PointsA[i] - data.PointsB[i]
    print(y)

    # for X
    i = 0
    for t in teams.Team:
        for r in data.index:
            if t == data['TeamA'][r]:
                X[r][i] = 1
            if t == data['TeamB'][r]:
                X[r][i] = -1
        i = i + 1
    print("original X\n", X)
    return X, y

def printRatingsTable(x):
    if x== 1:
        RatingsTable = pd.DataFrame({"team": teams.Team, "rating": R}).sort_values(by='rating',ascending=False)
        print("\n Ratings Table:\n", RatingsTable)
        return RatingsTable
    if x==2:
        s = int(len(R) / 2)
        off = [0] * s
        deff = [0] * s
        total = [0] * s
        i = 0
        flag = True
        for r in R:
            total[i] += r
            if flag:
                off[i] = r
                flag = False
            else:
                deff[i] = r
                i += 1
                flag = True

        RatingsTable = pd.DataFrame({"team": teams.Team, "Offense": off, "Defense": deff, "rating": total}).sort_values(by='rating', ascending=False)
        print("\n Ratings Table:\n", RatingsTable)
        return RatingsTable

def setSystemOfEquations(x,gof):
    # 1 for just basic least squares r
    # 2 to split rating into offence and defence
    if(x==1):
        return simpleLeastSquares()
    if(x==2):
        if(gof == 0 or gof == 1):
            return calcOffDef2(gof)
        else:
            return calcOffDef()

def GameOutcomeFunction(x, arr):
    from scipy.stats import norm
    mean = np.mean(arr)
    std = pow((50* np.sum(arr)),0.25)
    dist = norm(mean, std)
    cdf = dist.cdf(x)
    return cdf


def strengthTable():
    teams["PointsFor"] = [[] for x in range(len(teams))]
    teams["PointsAgainst"] = [[] for x in range(len(teams))]

    for g in data.index:
        A = data['TeamA'][g]
        B = data['TeamB'][g]
        indexA = np.where(teams.Team == A)[0][0]
        indexB = np.where(teams.Team == B)[0][0]
        #print(A, data['PointsA'][g], teams.PointsFor[indexA])
        teams.PointsFor[indexA].append(data['PointsA'][g])
        teams.PointsAgainst[indexA].append(data['PointsB'][g])
        teams.PointsFor[indexB].append(data['PointsB'][g])
        teams.PointsAgainst[indexB].append(data['PointsA'][g])
    #print(teams)
    return
def non_zero(x):
    if x==0:
        return 0.9
    else:
        return x

def recordTable(ratings):
    ratings["Won"] = [0 for x in range(len(ratings))]
    ratings["Lost"] = [0 for x in range(len(ratings))]
    ratings["PredictedWin"] = [0 for x in range(len(ratings))]
    ratings["PredictedLosses"] = [0 for x in range(len(ratings))]

    for g in data.index:
        A = data['TeamA'][g]
        B = data['TeamB'][g]
        indexA = np.where(ratings.team == A)[0][0]
        indexB = np.where(ratings.team == B)[0][0]
        #print(A, data['PointsA'][g], teams.PointsFor[indexA])
        ratings.at[indexA, 'Won']+=1
        ratings.at[indexB, 'Lost']+=1
        rA = ratings['rating'].values[ratings['team'] == A][0]
        rB = ratings['rating'].values[ratings['team'] == B][0]
        if (rA>rB):
            ratings.at[indexA, 'PredictedWin'] += 1
            ratings.at[indexB, 'PredictedLosses'] += 1
        elif (rA<rB):
            ratings.at[indexB, 'PredictedWin'] += 1
            ratings.at[indexA, 'PredictedLosses'] += 1
    ratings['PredictedWin'] = ratings['PredictedWin'].map(lambda x: non_zero(x))
    ratings['percentWin'] = ratings.apply(lambda row: row.Won/(row.Won + row.Lost), axis=1)
    ratings['predictedOutcome'] = ratings.apply(lambda row: row.PredictedWin / (row.PredictedWin + row.PredictedLosses), axis=1)
    ratings['BayesianFactor'] = ratings.apply(lambda row: row.percentWin / row.predictedOutcome, axis=1)
    print(ratings)
    return

def bayesianCorrection(RatingsTable):
    min = abs(RatingsTable['rating'].min())
    RatingsTable['rating'] = RatingsTable['rating'].map(lambda r: (r+min+0.1)*5)
    RatingsTable['Bayesian_corrected_Rating'] = RatingsTable.apply(lambda row: row.rating * row.BayesianFactor, axis=1)
    RatingsTable['Bayesian_correction_change'] = RatingsTable.apply(lambda row: row.Bayesian_corrected_Rating - row.rating, axis=1)
    RatingsTable =RatingsTable.sort_values(by='Bayesian_corrected_Rating', ascending=False)
    print(tabulate(RatingsTable,headers=RatingsTable.columns, tablefmt='fancy_grid'))
    return RatingsTable

def testingGOF():
    t = pd.DataFrame({'PointsA': [30,10,27,27,50,10,30,45,45,30,56],
                    'PointsB': [29,9,24,20,40,0,14,21,14,0,3],
                    'MasseyGOF(pA,pB)': [0.527,0.5359,0.5836,0.6924,0.7292,0.8548,0.8786, 0.9433,0.9823,0.9920, 0.9998]})
    t['myGOF(pA,pB)'] = t.apply(lambda r: GameOutcomeFunction(r.PointsA, [r.PointsA, r.PointsB]), axis = 1)
    print(t)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    rankType = 2 # 0=solve for rating, 2=solve for offense and defense rating
    dataset = 3 #0=exampleset, 1=national, 2=american, 3=all pro
    GOF=  1## 0=1 point for winning 1= normal distribution prob  2=none

    data, teams = readData(dataset)
    #strengthTable()
    #print(teams.shape)
    X,y = setSystemOfEquations(rankType, GOF)
    #print("X:\n", X, "\ny:\n", y, "\n")

    # calculating X transpose
    Xt = X.transpose()
    #print(Xt)

    # making the system of linear equations in the following form:
    # Mr=Y
    # where M=(Xt)*X
    #    and Y=(Xt)*y
    M = np.dot(Xt, X)
    #print(Xt, y)
    Y = np.dot(Xt, y)
    #print(M)
    #print(Y)

    # making sure all the ratings add up to 0
    # setting the last row OF m TO 1's
    for i in range(len(M[-1])):
        M[-1][i] = 1

    # setting the last row of Y to 0
    Y[-1] = 0
    #print("The system of linear equations to solve is:")
    #print("M=", M)
    #print("\nY=", Y)

    # solving the system of linear equations
    R = np.linalg.solve(M, Y)
    #print("R:\n", R)

    # printing team ratings in order
    RatingsTable = printRatingsTable(rankType)
    recordTable(RatingsTable)
    bayesianCorrection(RatingsTable)
