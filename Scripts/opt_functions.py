#Install Dependencies
import pandas as pd
import numpy as np
from scipy import linalg
import gurobipy as gp
from gurobipy import *


#############################################################################################################################################################################################
def gen_same_team_cor(data, year, week):
    
    """
    Generates relevaont optimization information for pairs of players on the same team
    Includes the correlation matrices and variances for optimization week and the season as well as the roster

    Parameters
    ----------
    data: Pandas Dataframe
        Dataframe that contains performance and prediction data for all the players
    year: int
        The season for which you plan to conduct the optimization
    week: int
        The week for which you plan to conduct the optimization

    Returns
    -------
    roster: list
        List of players who are elligible for lineup optimization
    roster_cor: NumPy matrix
        Same team correlation matrix up to the optimization week for the roster 
    roster_var: list
        List of variances up to the optimization week for the roster
    sim_cor: NumPy matrix
        Same team correlation matrix over the entire season for the roster. Used for simulation
    sim_var: list
        List of variances over the entire season for the roster. Used for simulation
    """

    possible_roster = data.loc[(data["Year"] == year) & (data["Week"] == week)]["Name"].unique()
    clipped_data = data.loc[(data["Year"] == year) & (data["Week"] < week) & (data["Name"].isin(possible_roster))]           #Data from weeks leading up to current week   
    sim_data = data.loc[(data["Year"] == year) & (data["Name"].isin(possible_roster))]
    roster = []                                                                       #to be filled with viable players
    roster_var = []                                                                   #How many players did we add
    roster_cor = np.zeros((0,0))                                                      #to be filled with covariances
    sim_var = []
    sim_cor = np.zeros((0,0))

    for team in data.loc[(data["Year"] == year) & (data["Week"] == week)]["Team2"].unique():
        
        #Create dataframe/pivot table from suitable roster
        output = pd.pivot_table(clipped_data.loc[clipped_data["Team2"] == team], index="Week", values="error", columns="Name")
        sim_output = pd.pivot_table(sim_data.loc[sim_data["Team2"] == team], index="Week", values="error", columns="Name")
        keep = output.columns[output.isna().sum()/output.shape[0] < 0.25]
        final_data = output[keep]
        final_sim_data = sim_output[keep]

        #Grow cov matrix by new roster
        new_dim = len(roster) + final_data.shape[1]
        ph_cor = np.copy(roster_cor)
        ph_sim_cor = np.copy(sim_cor)
        roster_cor = np.zeros((new_dim, new_dim))
        sim_cor = np.zeros((new_dim, new_dim))
        roster_cor[:len(roster), :len(roster)] = ph_cor
        sim_cor[:len(roster), :len(roster)] = ph_sim_cor

        #Add to roster
        roster.extend(final_data.columns)
        roster_var.extend([0]*final_data.shape[1])
        sim_var.extend([0]*final_data.shape[1])

        #Calculating variances and covariances
        for i in range(final_data.shape[1]):
            var = np.var(final_data[final_data.columns[i]].dropna())      #variance of player1
            roster_var[roster.index(final_data.columns[i])] = var

            var_sim = np.var(final_sim_data[final_data.columns[i]].dropna())      #variance of player1
            sim_var[roster.index(final_data.columns[i])] = var_sim

            for j in range(i+1, final_data.shape[1]):
                
                #Generate covariances and correlations from paired data
                pair = final_data[[final_data.columns[i], final_data.columns[j]]] 
                pair = pair.dropna()
                pair_cor = np.corrcoef(pair.iloc[:,0], pair.iloc[:,1])     #correlation for the pair

                sim_pair = final_sim_data[[final_data.columns[i], final_data.columns[j]]] 
                sim_pair = sim_pair.dropna()
                sim_pair_cor = np.corrcoef(sim_pair.iloc[:,0], sim_pair.iloc[:,1])     #correlation for the pair
                

                #Assign covariance to proper location in the matrix
                roster_cor[roster.index(final_data.columns[i]), roster.index(final_data.columns[i])]  = 1
                roster_cor[roster.index(final_data.columns[j]), roster.index(final_data.columns[j])]  = 1
                roster_cor[roster.index(final_data.columns[i]), roster.index(final_data.columns[j])] = pair_cor[0,1]
                roster_cor[roster.index(final_data.columns[j]), roster.index(final_data.columns[i])] = pair_cor[0,1]

                sim_cor[roster.index(final_data.columns[i]), roster.index(final_data.columns[i])]  = 1
                sim_cor[roster.index(final_data.columns[j]), roster.index(final_data.columns[j])]  = 1
                sim_cor[roster.index(final_data.columns[i]), roster.index(final_data.columns[j])] = sim_pair_cor[0,1]
                sim_cor[roster.index(final_data.columns[j]), roster.index(final_data.columns[i])] = sim_pair_cor[0,1]
    
    return roster, roster_cor, roster_var, sim_cor, sim_var
#############################################################################################################################################################################################

#############################################################################################################################################################################################
def gen_opp_team_cor(data, same_cor, roster, year, week):
    
    """
    Updates same team correlation matrix with correlations for pairs of competing players up to the optimization week

    Parameters
    ----------
    data: Pandas Dataframe
        Dataframe that contains performance and prediction data for all the players
    same_cor: NumPy matrix
        Same team correlation up to optimization week
    roster: list
        List of lligible players for optimization week
    year: int
        The season for which you plan to conduct the optimization
    week: int
        The week for which you plan to conduct the optimization   

    Returns
    -------
    init_cor: NumPy matrix
        Correlation matrix up to the optimization week which includes correlations for pairs of competing players 
    """

    #Create initial Correlation Matrix
    init_cor = np.copy(same_cor)

    for player in roster:

        #Collect relevant data for player1
        clipped_data = data.loc[(data["Year"] == year) & (data["Week"] < week) & (data["Name"].isin(roster))]
        play1 = clipped_data.loc[clipped_data["Name"] == player]

        #Initialize plaeyer vs. position error data
        qb_err = np.empty((0, 2))
        wr_err = np.empty((0, 2))
        rb_err = np.empty((0, 2))
        te_err = np.empty((0, 2))
        dst_err = np.empty((0, 2))

        #Begin calculating player vs. position correlations from prior week's data
        for i in range(play1.shape[0]):
            wek = play1["Week"].iloc[i]
            oppt = play1["Oppt"].iloc[i]
            play1_error = play1["error"].iloc[i]
            others = clipped_data.loc[(clipped_data["Week"] == wek) & (clipped_data["Team2"] == oppt)]          #competing player data for wek

            for pos in others["Pos"].unique():

                #Collect vector of errors for each position. Repeat player1 errors for the number of players at the position. Create error pairs for position
                pos_error = np.expand_dims(others.loc[others["Pos"] == pos]["error"], axis=1)
                play1_errs = np.expand_dims(np.repeat(play1_error, pos_error.shape[0]), axis=1)
                error_pairs = np.concatenate((play1_errs, pos_error), axis = 1)

                #Depending on the current player2 position, append error pair to position data
                if pos == "QB":
                    qb_err = np.concatenate((qb_err, error_pairs), axis=0)
                if pos == "WR":
                    wr_err = np.concatenate((wr_err, error_pairs), axis=0)
                if pos == "RB":
                    rb_err = np.concatenate((rb_err, error_pairs), axis=0)
                if pos == "TE":
                    te_err = np.concatenate((te_err, error_pairs), axis=0)
                if pos == "DST":
                    dst_err = np.concatenate((dst_err, error_pairs), axis=0)

        #Generate player1 vs. position correlations
        qb_cor = np.corrcoef(qb_err[:, 0], qb_err[:, 1])
        wr_cor = np.corrcoef(wr_err[:, 0], wr_err[:, 1])
        rb_cor = np.corrcoef(rb_err[:, 0], rb_err[:, 1])
        te_cor = np.corrcoef(te_err[:, 0], te_err[:, 1])
        dst_cor = np.corrcoef(dst_err[:, 0], dst_err[:, 1])

        #Begin populating player1 vs player2 correlations for optimization week
        team2 = data.loc[(data["Name"] == player) & (data["Week"] == week)]["Oppt"].iloc[0]
        
        for play2 in data.loc[(data["Name"].isin(roster)) & (data["Week"] == week) & (data["Team2"] == team2)]["Name"].unique():
            
            # Locate player2 position and fetch position correlation for player1 vs. position
            # If value has not been touched, populate with current correlation
            # If value has been touched, use lower magnitude correlation
            if data.loc[(data["Week"] == week) & (data["Name"] == play2)]["Pos"].iloc[0] == "QB":
                if init_cor[roster.index(player), roster.index(play2)] == 0:
                    init_cor[roster.index(player), roster.index(play2)] = qb_cor[0,1]
                    init_cor[roster.index(play2), roster.index(player)] = qb_cor[0,1]
                else:
                    if abs(qb_cor[0,1]) < abs(init_cor[roster.index(player), roster.index(play2)]):
                        init_cor[roster.index(player), roster.index(play2)] = qb_cor[0,1]
                        init_cor[roster.index(play2), roster.index(player)] = qb_cor[0,1]                    
            if data.loc[(data["Week"] == week) & (data["Name"] == play2)]["Pos"].iloc[0] == "WR":
                if init_cor[roster.index(player), roster.index(play2)] == 0:
                    init_cor[roster.index(player), roster.index(play2)] = wr_cor[0,1]
                    init_cor[roster.index(play2), roster.index(player)] = wr_cor[0,1]
                else:
                    if abs(wr_cor[0,1]) < abs(init_cor[roster.index(player), roster.index(play2)]):
                        init_cor[roster.index(player), roster.index(play2)] = wr_cor[0,1]
                        init_cor[roster.index(play2), roster.index(player)] = wr_cor[0,1] 
            if data.loc[(data["Week"] == week) & (data["Name"] == play2)]["Pos"].iloc[0] == "RB":
                if init_cor[roster.index(player), roster.index(play2)] == 0:
                    init_cor[roster.index(player), roster.index(play2)] = rb_cor[0,1]
                    init_cor[roster.index(play2), roster.index(player)] = rb_cor[0,1]
                else:
                    if abs(rb_cor[0,1]) < abs(init_cor[roster.index(player), roster.index(play2)]):
                        init_cor[roster.index(player), roster.index(play2)] = rb_cor[0,1]
                        init_cor[roster.index(play2), roster.index(player)] = rb_cor[0,1] 
            if data.loc[(data["Week"] == week) & (data["Name"] == play2)]["Pos"].iloc[0] == "TE":
                if init_cor[roster.index(player), roster.index(play2)] == 0:
                    init_cor[roster.index(player), roster.index(play2)] = te_cor[0,1]
                    init_cor[roster.index(play2), roster.index(player)] = te_cor[0,1]
                else:
                    if abs(te_cor[0,1]) < abs(init_cor[roster.index(player), roster.index(play2)]):
                        init_cor[roster.index(player), roster.index(play2)] = te_cor[0,1]
                        init_cor[roster.index(play2), roster.index(player)] = te_cor[0,1] 
            if data.loc[(data["Week"] == week) & (data["Name"] == play2)]["Pos"].iloc[0] == "DST":
                if init_cor[roster.index(player), roster.index(play2)] == 0:
                    init_cor[roster.index(player), roster.index(play2)] = dst_cor[0,1]
                    init_cor[roster.index(play2), roster.index(player)] = dst_cor[0,1]
                else:
                    if abs(dst_cor[0,1]) < abs(init_cor[roster.index(player), roster.index(play2)]):
                        init_cor[roster.index(player), roster.index(play2)] = dst_cor[0,1]
                        init_cor[roster.index(play2), roster.index(player)] = dst_cor[0,1]
    return init_cor 
#############################################################################################################################################################################################

#############################################################################################################################################################################################
def gen_opp_team_cor_sim(data, same_cor, roster, year, week):
    
    """
    Updates same team correlation matrix with correlations for pairs of competing players for the entire season
    Used for simulation

    Parameters
    ----------
    data: Pandas Dataframe
        Dataframe that contains performance and prediction data for all the players
    same_cor: NumPy matrix
        Same team correlation up to optimization week
    roster: list
        List of lligible players for optimization week
    year: int
        The season for which you plan to conduct the optimization
    week: int
        The week for which you plan to conduct the optimization   

    Returns
    -------
    init_cor: NumPy matrix
        Correlation matrix for the entire season which includes correlations for pairs of competing players   
    """

    #Create initial Correlation Matrix
    init_cor = np.copy(same_cor)
    
    for player in roster:

        #Collect relevant data for player1
        clipped_data = data.loc[(data["Year"] == year) & (data["Name"].isin(roster))]        
        play1 = clipped_data.loc[clipped_data["Name"] == player]                             

        #Initialize plaeyer vs. position error data
        qb_err = np.empty((0, 2))
        wr_err = np.empty((0, 2))
        rb_err = np.empty((0, 2))
        te_err = np.empty((0, 2))
        dst_err = np.empty((0, 2))

        #Begin calculating player vs. position correlations from prior week's data
        for i in range(play1.shape[0]):

            wek = play1["Week"].iloc[i]
            oppt = play1["Oppt"].iloc[i]
            play1_error = play1["error"].iloc[i]
            others = clipped_data.loc[(clipped_data["Week"] == wek) & (clipped_data["Team2"] == oppt)]     #competing player data for wek

            for pos in others["Pos"].unique():

                #Collect vector of errors for each position. Repeat player1 errors for the number of players at the position. Create error pairs for position
                pos_error = np.expand_dims(others.loc[others["Pos"] == pos]["error"], axis=1)
                play1_errs = np.expand_dims(np.repeat(play1_error, pos_error.shape[0]), axis=1)
                error_pairs = np.concatenate((play1_errs, pos_error), axis = 1)

                #Depending on the current player2 position, append error pair to position data
                if pos == "QB":
                    qb_err = np.concatenate((qb_err, error_pairs), axis=0)
                if pos == "WR":
                    wr_err = np.concatenate((wr_err, error_pairs), axis=0)
                if pos == "RB":
                    rb_err = np.concatenate((rb_err, error_pairs), axis=0)
                if pos == "TE":
                    te_err = np.concatenate((te_err, error_pairs), axis=0)
                if pos == "DST":
                    dst_err = np.concatenate((dst_err, error_pairs), axis=0)
        
        #Generate player1 vs. position correlations
        qb_cor = np.corrcoef(qb_err[:, 0], qb_err[:, 1])
        wr_cor = np.corrcoef(wr_err[:, 0], wr_err[:, 1])
        rb_cor = np.corrcoef(rb_err[:, 0], rb_err[:, 1])
        te_cor = np.corrcoef(te_err[:, 0], te_err[:, 1])
        dst_cor = np.corrcoef(dst_err[:, 0], dst_err[:, 1])

        #Begin populating player1 vs player2 correlations for optimization week
        team2 = data.loc[(data["Name"] == player) & (data["Week"] == week)]["Oppt"].iloc[0]
        
        for play2 in data.loc[(data["Name"].isin(roster)) & (data["Week"] == week) & (data["Team2"] == team2)]["Name"].unique():

            # Locate player2 position and fetch position correlation for player1 vs. position
            # If value has not been touched, populate with current correlation
            # If value has been touched, use lower magnitude correlation
            if data.loc[(data["Week"] == week) & (data["Name"] == play2)]["Pos"].iloc[0] == "QB":
                if init_cor[roster.index(player), roster.index(play2)] == 0:
                    init_cor[roster.index(player), roster.index(play2)] = qb_cor[0,1]
                    init_cor[roster.index(play2), roster.index(player)] = qb_cor[0,1]
                else:
                    if abs(qb_cor[0,1]) < abs(init_cor[roster.index(player), roster.index(play2)]):
                        init_cor[roster.index(player), roster.index(play2)] = qb_cor[0,1]
                        init_cor[roster.index(play2), roster.index(player)] = qb_cor[0,1]                    
            if data.loc[(data["Week"] == week) & (data["Name"] == play2)]["Pos"].iloc[0] == "WR":
                if init_cor[roster.index(player), roster.index(play2)] == 0:
                    init_cor[roster.index(player), roster.index(play2)] = wr_cor[0,1]
                    init_cor[roster.index(play2), roster.index(player)] = wr_cor[0,1]
                else:
                    if abs(wr_cor[0,1]) < abs(init_cor[roster.index(player), roster.index(play2)]):
                        init_cor[roster.index(player), roster.index(play2)] = wr_cor[0,1]
                        init_cor[roster.index(play2), roster.index(player)] = wr_cor[0,1] 
            if data.loc[(data["Week"] == week) & (data["Name"] == play2)]["Pos"].iloc[0] == "RB":
                if init_cor[roster.index(player), roster.index(play2)] == 0:
                    init_cor[roster.index(player), roster.index(play2)] = rb_cor[0,1]
                    init_cor[roster.index(play2), roster.index(player)] = rb_cor[0,1]
                else:
                    if abs(rb_cor[0,1]) < abs(init_cor[roster.index(player), roster.index(play2)]):
                        init_cor[roster.index(player), roster.index(play2)] = rb_cor[0,1]
                        init_cor[roster.index(play2), roster.index(player)] = rb_cor[0,1] 
            if data.loc[(data["Week"] == week) & (data["Name"] == play2)]["Pos"].iloc[0] == "TE":
                if init_cor[roster.index(player), roster.index(play2)] == 0:
                    init_cor[roster.index(player), roster.index(play2)] = te_cor[0,1]
                    init_cor[roster.index(play2), roster.index(player)] = te_cor[0,1]
                else:
                    if abs(te_cor[0,1]) < abs(init_cor[roster.index(player), roster.index(play2)]):
                        init_cor[roster.index(player), roster.index(play2)] = te_cor[0,1]
                        init_cor[roster.index(play2), roster.index(player)] = te_cor[0,1] 
            if data.loc[(data["Week"] == week) & (data["Name"] == play2)]["Pos"].iloc[0] == "DST":
                if init_cor[roster.index(player), roster.index(play2)] == 0:
                    init_cor[roster.index(player), roster.index(play2)] = dst_cor[0,1]
                    init_cor[roster.index(play2), roster.index(player)] = dst_cor[0,1]
                else:
                    if abs(dst_cor[0,1]) < abs(init_cor[roster.index(player), roster.index(play2)]):
                        init_cor[roster.index(player), roster.index(play2)] = dst_cor[0,1]
                        init_cor[roster.index(play2), roster.index(player)] = dst_cor[0,1]
    return init_cor
#############################################################################################################################################################################################

#############################################################################################################################################################################################
def get_projections(data, roster, year, week):

    """
    Fetch projections for the optimization week

    Parameters
    ----------
    data: Pandas Dataframe
        Dataframe that contains performance and prediction data for all the players
    roster: list
        List of lligible players for optimization week
    year: int
        The season for which you plan to conduct the optimization
    week: int
        The week for which you plan to conduct the optimization     

    Returns
    -------
    projections: list
        List of projections for the optimization week 
    """

    projections = []
    for player in roster:
        projections.append(data.loc[(data["Year"] == year) & (data["Week"] == week) & (data["Name"] == player)]["ProjDKPts"].iloc[0])
    
    return projections
#############################################################################################################################################################################################

#############################################################################################################################################################################################
def get_costs(data, roster, year, week):

    """
    Fetch costs for the optimization week

    Parameters
    ----------
    data: Pandas Dataframe
        Dataframe that contains performance and prediction data for all the players
    roster: list
        List of lligible players for optimization week
    year: int
        The season for which you plan to conduct the optimization
    week: int
        The week for which you plan to conduct the optimization     

    Returns
    -------
    costs: list
        List of costs for the optimization week 
    """

    costs = []
    for player in roster:
        costs.append(data.loc[(data["Year"] == year) & (data["Week"] == week) & (data["Name"] == player)]["DK salary"].iloc[0])
    
    return costs
#############################################################################################################################################################################################

#############################################################################################################################################################################################
def get_points(data, roster, year, week):

    """
    Fetch actual points for the optimization week

    Parameters
    ----------
    data: Pandas Dataframe
        Dataframe that contains performance and prediction data for all the players
    roster: list
        List of lligible players for optimization week
    year: int
        The season for which you plan to conduct the optimization
    week: int
        The week for which you plan to conduct the optimization     

    Returns
    -------
    points: list
        List of actual points for the optimization week 
    """
    points = []
    for player in roster:
        points.append(data.loc[(data["Year"] == year) & (data["Week"] == week) & (data["Name"] == player)]["DK points"].iloc[0])
    
    return points
#############################################################################################################################################################################################

#############################################################################################################################################################################################
def get_errors(data, roster, year, week):

    """
    Fetch errors for the optimization week

    Parameters
    ----------
    data: Pandas Dataframe
        Dataframe that contains performance and prediction data for all the players
    roster: list
        List of lligible players for optimization week
    year: int
        The season for which you plan to conduct the optimization
    week: int
        The week for which you plan to conduct the optimization     

    Returns
    -------
    errors: list
        List of errors for the optimization week 
    """

    errors = []
    for player in roster:
        errors.append(data.loc[(data["Year"] == year) & (data["Week"] == week) & (data["Name"] == player)]["error"].iloc[0])
    
    return errors
#############################################################################################################################################################################################

#############################################################################################################################################################################################
def get_pos_indices(data, roster, year, week):

    """
    Find the indices in the roster list that corresponds to each of the positions

    Parameters
    ----------
    data: Pandas Dataframe
        Dataframe that contains performance and prediction data for all the players
    roster: list
        List of lligible players for optimization week
    year: int
        The season for which you plan to conduct the optimization
    week: int
        The week for which you plan to conduct the optimization     

    Returns
    -------
    pos_indices: dictionary
        Dictionary containing lists of indexes corresponding to each of the 5 positions
    """

    pos_indices = {}
    pos_indices["QB"] = []
    pos_indices["WR"] = []
    pos_indices["RB"] = []
    pos_indices["TE"] = []
    pos_indices["DST"] = []
    
    for i in range(len(roster)):
        pos = data.loc[(data["Year"] == year) & (data["Week"] == week) & (data["Name"] == roster[i])]["Pos"].iloc[0]

        if pos == "QB":
            pos_indices["QB"].append(i)
        if pos == "WR":
            pos_indices["WR"].append(i)
        if pos == "RB":
            pos_indices["RB"].append(i)
        if pos == "TE":
            pos_indices["TE"].append(i)
        if pos == "DST":
            pos_indices["DST"].append(i)
    
    return pos_indices
#############################################################################################################################################################################################

#############################################################################################################################################################################################
def return_names(roster, solution):

    """
    Fetch player names of the solution based on the raw binary solution vector

    Parameters
    ----------
    roster: list
        List of elligible players for optimization week
    solution: list
        Raw binary solution vector from optimization output

    Returns
    -------
    lineup: list
        List of players selected to the lineup 
    """

    lineup = []
    for i in range(len(solution)):
        if solution[i] > 0.5:
            lineup.append(roster[i])
    
    return lineup
#############################################################################################################################################################################################

#############################################################################################################################################################################################
def MILP(projections, costs, pos_dict, exclude):

    """
    Fetch costs for the optimization week

    Parameters
    ----------
    projections: list
        List of projections for the optimization week 
    costs: list
        List of costs for the optimization week 
    pos_dict: dictionary
        Dictionary containing lists of indexes corresponding to each of the 5 positions
    exclude: list
        List of indices/players that should be excluded from the optimization    

    Returns
    -------
    obj_value: list
        List of costs for the optimization week 
    x.X: list
        Raw binary decision vector for player selection to lineup
    """

    model = gp.Model()

    #Crate binary selection variable
    x = model.addMVar(len(projections), vtype=GRB.BINARY)

    #Budget Constraint
    model.addConstr(sum([costs[i]*x[i] for i in range(len(costs))]) <= 50000)

    #Lineup Constraints
    model.addConstr(sum([x[i] for i in pos_dict["QB"]]) == 1)      #QB constraint
    model.addConstr(sum([x[i] for i in pos_dict["WR"]]) >= 3)      #WR constraint
    model.addConstr(sum([x[i] for i in pos_dict["RB"]]) >= 2)      #RB constraint
    model.addConstr(sum([x[i] for i in pos_dict["TE"]]) >= 1)      #TE constraint
    model.addConstr(sum([x[i] for i in pos_dict["DST"]]) == 1)     #DST constraint
    model.addConstr(sum(x) == 9)                                   #QB constraint

    #Exclusion constraints
    for index in exclude:
        model.addConstr(x[index] == 0)

    #Add Objective
    model.setObjective(sum([projections[i]*x[i] for i in range(len(projections))]), GRB.MAXIMIZE)

    #Solving
    model.optimize()
    obj_value = model.getObjective().getValue()

    return obj_value, x.X
#############################################################################################################################################################################################

#############################################################################################################################################################################################
def Robust(projections, costs, pos_dict, exclude, cov_matrix, rho, shape):
    
    """
    Fetch costs for the optimization week

    Parameters
    ----------
    projections: list
        List of projections for the optimization week 
    costs: list
        List of costs for the optimization week 
    pos_dict: dictionary
        Dictionary containing lists of indexes corresponding to each of the 5 positions
    exclude: list
        List of indices/players that should be excluded from the optimization
    cov_matrix: NumPy array
        Covariance matrix of the errors up to the optimization week
    rho: float
        Uncertainty set size
    shape: string
        Uncertainty set shape
        MUST BE "box", "ball", or "polygon"    

    Returns
    -------
    obj_value: list
        List of costs for the optimization week 
    x.X: list
        Raw binary decision vector for player selection to lineup
    """
    
    #Calculate coefficients for Uncertainty Set
    M = linalg.sqrtm(np.linalg.inv(cov_matrix))
    coef = np.linalg.inv(M).T

    #Instantiate the optimizer
    model = gp.Model()
    model.Params.NumericFocus = 3

    #Crate binary selection variable
    x = model.addMVar(len(projections), vtype=GRB.BINARY)

    #Budget Constraint
    model.addConstr(sum([costs[i]*x[i] for i in range(len(costs))]) <= 50000)


    #Lineup Constraints
    model.addConstr(sum([x[i] for i in pos_dict["QB"]]) == 1)      #QB constraint
    model.addConstr(sum([x[i] for i in pos_dict["WR"]]) >= 3)      #WR constraint
    model.addConstr(sum([x[i] for i in pos_dict["RB"]]) >= 2)      #RB constraint
    model.addConstr(sum([x[i] for i in pos_dict["TE"]]) >= 1)      #TE constraint
    model.addConstr(sum([x[i] for i in pos_dict["DST"]]) == 1)     #DST constraint
    model.addConstr(sum(x) == 9)                                   #QB constraint

    #Exclusion constraints
    for index in exclude:
        model.addConstr(x[index] == 0)

    #Robust Constraints
    safety = model.addVar()
    error_vec = model.addMVar(len(projections), lb=-GRB.INFINITY)    
    model.addConstr(error_vec == coef @ x)

    #Selecting proper norm from uncertainty set shape
    if shape == "box":
        vec_norm = 1
    if shape == "ball":
        vec_norm = 2
    if shape == "polygon":
        vec_norm = GRB.INFINITY
    
    model.addGenConstrNorm(safety, error_vec, vec_norm)

    #Add Objective
    model.setObjective(sum([projections[i]*x[i] for i in range(len(projections))]) -  rho * safety, GRB.MAXIMIZE)

    #Solving
    model.optimize()
    obj_value = model.getObjective().getValue()

    return obj_value, x.X
#############################################################################################################################################################################################



