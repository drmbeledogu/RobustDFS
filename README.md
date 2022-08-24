# RobustDFS
Robust Optimization framework for Daily Fantasy Football

### Description
The goal of daily fantasy sports is to draft a proper lineup $(x∈\\{0,1\\}^N)$ for which each player has a cost $(c∈R_+^N)$, subject to a budget constraint, that maximizes the number of points you will receive. Given that you have projections for what each player will score $(p∈R^N)$, this problem can be formulated as a simple mixed integer linear programming problem (MILP):
$$\max_{x} \\; p^Tx$$
$$s.t. \quad c^Tx \leq Budget$$
At first glance, this seems fair however the payout structure of the competition is being neglected. Secondly player projections are rarely correct and the errors around these projections are uncertain. Given the variance of projection error and the competition payout structure, is maximizing projected points the correct strategy?

In this work, focus was placed on 50/50 and Double-Up competitions where the top percentage, 50% and 45% respectively, all receive the same payout, and the bottom percentage receive no payout. Now imagine lineup 1 has a higher mean but wider variance in performance than lineup 2. Although lineup 2 performs worse on average, if the low side of the probability distribution is still above the payout line, you would select lineup 2 anyways because the expected value of the payout is still higher than lineup 1. In a sense we are looking for the lineup that maximizes the low side of the probability distribution or “maximizes the worst-case scenario".

![alt text](https://github.com/drmbeledogu/RobustDFS/raw/main/Documents/example_lineup_comparison.jpg)

There is an optimization paradigm that aims to “maximize the worst-case scenario” called Robust Optimization. In 50/50 and Double-Up, it may be advantageous to maximize the worst possible performance of your lineup given some uncertainty $U$ set around player performance. The robust formulation is:

$$\max_{x} \\; p^Tx-\rho\\|\Sigma^\frac{1}{2}x\\|$$

$$s.t. \quad c^Tx \leq Budget$$

This repository aims to investigate these ideas and generate lineups using both optimization frameworks. Details around the construction of uncertainty sets as well as the derivation of the Robust formulation will be included in a paper that will soon be added to the respository.

### Dependencies
* [NumPy](https://numpy.org/install/)
* [SciPy](https://scipy.org/install/)
* [Pandas](https://pandas.pydata.org/docs/getting_started/install.html)
* [Matplotlib](https://matplotlib.org/stable/users/installing/index.html)
* [Distfit](https://erdogant.github.io/distfit/pages/html/Installation.html)
* [Gurobipy](https://www.gurobi.com/documentation/9.5/quickstart_mac/cs_python_installation_opt.html): _Gurobipy requires a Gurobi License to solve the Robust DFS problem. See instructions [here]() to obtain a license._

### Data
The file, `udfs_data2021.csv`, is a combination of historical Draftkings point production and salary for every player during the 2021 season scraped from [rotoguru1.com](http://rotoguru1.com/cgi-bin/fyday.pl?gameyr=dk2021) and DraftKings/Fanduel projections provided by Caleb Nelson at [dfsforecast.com](https://dfsforecast.com/). Descriptions of each of the fields are listed below:
* `Year (int)`: Year of the start of the football season
* `Week (int)`: Week within the season that the game was played
* `Name (string)`: Player Name
* `Pos (string)`: Player position
* `Team (string)`: Team for which the player played for
* `ProjDKPts (float)`: Projected Draftkings points
* `ProjFDPts (float)`: Projected FanDueal points
* `Team2 (string)`: Alternate team respresentation for "Team" field
* `Oppt (string)`: Opponent's team name
* `DK points (float)`: Actual Draftkings points
* `DK salary (float)`: Actual Draftkings salary
* `error (float)`: Difference between Draftkings projected points and actual points | `DK points - ProjDKPts`

### Supporting Python Files
* `nearest_correlation.py`: This file contains the functions required to solve the nearrest correlation matrix problem. This implementation uses an alternating projections algorithm developed by Nick Higham. Details about the fundamentals of Nick's alternating projections method can be found [here](https://www.maths.manchester.ac.uk/~higham/narep/narep369.pdf). The python implementation was developed by Mike Croucher and the original repo with details are [here](https://github.com/mikecroucher/nearest_correlation). More details on the reason for why this file is needed will come with the research paper.
* `opt_functions.py`: This file contains all the helper functions required to generate relevant information to solve the mixed integer linear programming and robust optimization problems.It also contains the functions that directly solves these two problems.

### Notebooks
* `production_scrape.ipynb`: This notebook is not required. It is what I used to scrape the historical production data from rotoguru1.com. I thought it may be useful to include for others to run this analysis on seasons other than 2021-2022 if they have projection data for those other seasons.
* `EDA.ipynb`: This notebook is used for exploratory data analysis, hence the norebook title. Each cell has different purposes listed below. The notebook itself will have more commentary/fundamental reasoning for why a particular attribute of the data is being explored/investigated.
  * **Cell 2:** View raw dataframe
  * **Cell 3:** Investigate distribution of errors for each position.
  * **Cells 4&5:** Investigate correlation of errors between positional pairs on the same team
  * **Cells 6&7:** Investigate correlation of errors between positional pairs on competing teams
  * **Cell 8:** Visualize the shape and size of uncertainty sets over simulated errors
* `optimize_lineup.ipynb`: This is the notebook used to actually create MILP and robust lineups and to investigate the results of these lineups. More details about the methdology/fundamental mathematics for each cell will soon be available in the notebook itself.
