# ID3-Decision-Tree-with-GraphViz

Made use of graphviz feature of sklearn module and hence in order to put it to good use and get the desired output, do as follows:

1. Install Graphviz2.38 and install it in C:\Program Files\ (This way, you would not have to set the enviorment variable as we have taken care of it in the code)
2. Run the code per steps


Details of the code:

1. Already set up the CSV file as per use(Instead of True and False elements we made use of YESs and NOs)
2. Split the data into target and feature sets that were to be used to traverse through the tree.
3. Vectored the data to get it in the form of array
4. Converted the data to be usable into the code i.e., use encoding to convert strings to float/integers 
5. Fir the tree to the DecisionTreeClassifier class to get the desired tree
6. Displayed the tree using graphviz 
