import pandas as pd
import numpy as np

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

def setSystemOfEquations(x):
    # 1 for just basic least squares r
    # 2 to split rating into offence and defence
    if(x==1):
        return simpleLeastSquares()
    if(x==2):
        return calcOffDef()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    rankType = 2 # 0=solve for rating, 2=solve for offense and defense rating
    dataset = 3 #0=exampleset, 1=national, 2=american, 3=all pro

    data, teams = readData(dataset)
    #print(teams.shape)
    X,y = setSystemOfEquations(rankType)
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
    M[-1]
    for i in range(len(M[-1])):
        M[-1][i] = 1

    # setting the last row of Y to 0
    Y[-1] = 0
    print("The system of linear equations to solve is:")
    print("M=", M)
    print("\nY=", Y)

    # solving the system of linear equations
    R = np.linalg.solve(M, Y)
    print("R:\n", R)

    # printing team ratings in order
    printRatingsTable(rankType)
