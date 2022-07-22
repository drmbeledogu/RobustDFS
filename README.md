# RobustDFS
Robust Optimization framework for Daily Fantasy Football

### Dependencies
* [NumPy](https://numpy.org/install/)
* [SciPy](https://scipy.org/install/)
* [Pandas](https://pandas.pydata.org/docs/getting_started/install.html)
* [Matplotlib](https://matplotlib.org/stable/users/installing/index.html)
* [Distfit](https://erdogant.github.io/distfit/pages/html/Installation.html)
* [Gurobipy](https://www.gurobi.com/documentation/9.5/quickstart_mac/cs_python_installation_opt.html): _Gurobipy requires a Gurobi License to solve the Robust DFS problem. See instructions [here]() to obtain a license._

### Data
The file, udfs_data2021.csv, is a combination of historical Draftkings point production and salary for every player during the 2021 season scraped from [rotoguru1.com](http://rotoguru1.com/cgi-bin/fyday.pl?gameyr=dk2021) and DraftKings/Fanduel projections provided by Caleb Nelson at [dfsforecast.com](https://dfsforecast.com/). Descriptions of each of the fields are listed below:
* **Year (int):** Year of the start of the football season
* **Week (int):** Week within the season that the game was played
* **Name (string):** Player Name
* **Pos (string):** Player position
* **Team (string):** Team for which the player played for
* **ProjDKPts (float):** Projected Draftkings points
* **ProjFDPts (float):** Projected FanDueal points
* **Team2 (string):** Alternate team respresentation for "Team" field
* **Oppt (string):** Opponent's team name
* **DK points (float):** Actual Draftkings points
* **DK salary (float):** Actual Draftkings salary
* **error (float):** Difference between Draftkings projected points and actual points | ProjDKPts - DK points

### Supporting Python Files

### Notebooks
