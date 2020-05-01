# Tolerance Details

PowerMethod and StochasticBatchPowerMethod can use any combination of three tolerance criteria to determine convergence.
1. Root Mean Squared Error of Top K Components (RMSE or RMSE_K)
2. Weighted Subspace Distance 
3. Singular Value "Q"-Convergence


## Root Mean Squared Error of Top K Components (RMSE or RMSE_K)
Let $USV$ be 