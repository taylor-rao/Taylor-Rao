#Capstone

The dataset I used was given to me by the statistics department who got it from the A&M baseball team.

My goal with this project was for coaches of A&M's baseball team to be able to use scouting reports to determine which of their batters to match up against which of the oposing team's pitchers.

I used pandas, numpy, matplotlib and scikit-learn to perform my analysis.

Going off of the ipynb file, the first cell (ln [2]) is where I load the data, load the packages, and subset the data to include only the rows where the batter is an A&M player.

After this, I showed what the rows of the data look like using the 'head' method. I identified 'pitchCall' as my response variable. In the cell titled (ln [5]) I looked at the response variable's different categories and the number of each occurance. I did the same thing with batters in the next cell.

I then looked at the number of null values of each variable and got rid of those with a high number of such values. I then looked at the remaining variables and selected only those that would make sense to include in a model.

Scikit-learn does not allow for building models with categorical data In the cell titled 'ln [14]' I used pandas to do something called 'one-hot encoding' which is where each categorical variable split into n boolean variabes where n is the number of categories the origional variable can take. 

In ln[14] - ln[24] I then split the data into testing and training data and tested the accuract of various machine learning algorithms for predicting 'pitchCall'. I observed the the tree based algorithms performed much better than the regression based ones.

My goal at this time was to build a model using a small subset of variables without sacrificing much in the way of accuracy. I then wrote a graph the used forward selection to select subsets of size 1 - 10 and display the variable selected and the accuracy of a model with that and all previous variables included. 



