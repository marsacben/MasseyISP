import pandas as pd
import numpy as np

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def finDir():
    import os
    for f in os.listdir("/"):
        print(f)

def readData(x):
    if x ==0:
        pathGames = 'Data/MasseyGameData - MasseyExampleData.csv'
        pathTeams ='Data/MasseyGameData - teamsExample.csv'

    if x==2:
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



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data, teams = readData(0)

    # instantiating matrices
    y = np.zeros(shape=(len(data.PointsA)))
    X = np.zeros(shape=(len(data.PointsA), len(teams)))
    #print(X, y)

    # setting matrix values based on data

    # for y
    for i in range(len(data.PointsA)):
        y[i] = data.PointsA[i] - data.PointsB[i]
    #print(y)

    # for X
    i = 0;
    for t in teams.Team:
        for r in data.index:
            if t == data['TeamA'][r]:
                X[r][i] = 1
            if t == data['TeamB'][r]:
                X[r][i] = -1
        i = i + 1
    #print(X)

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
    print("The system of lenear equations to solve is:")
    print("M=", M)
    print("\nY=", Y)

    # solving the system of linear equations
    R = np.linalg.solve(M, Y)
    #print(R)

    # printing team ratings in order
    RatingsTable = pd.DataFrame({"team": teams.Team, "rating": R}).sort_values(by='rating', ascending=False)
    print("\n Ratings Table:\n", RatingsTable)
