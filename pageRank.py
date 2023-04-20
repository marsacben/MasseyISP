import pandas as pd
import numpy as np
import networkx as nx

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

def makematrix():
    # instantiating matrices

    G = nx.Graph()
    for t in teams.Team:
        G.add_node(t)
    for r in data.index:
        pointDiff = data['PointsA'][r] - data['PointsB'][r]
        G.add_weighted_edges_from([(data['TeamB'][r], data['TeamA'][r], pointDiff)])

    return G


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataset = 1 #0=exampleset, 1=national, 2=american, 3=all pro
    data, teams = readData(dataset)
    G = makematrix()
    print(G)
    print(nx.pagerank(G, alpha=0.9))


