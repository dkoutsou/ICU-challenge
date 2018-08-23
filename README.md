# ICU Challenge
In this project we classified on 7 tasks using an XGBoost Classifier and also printed the feature importance so as to see which features help in the classification.

Instructions on how to run the project. 

* Create a virtualenvironment and install requirements.txt
* Example usage of an argparser

	```
	-> % python code/main.py -h
	usage: main.py [-h] [-f {tune,train_eval}] [-d DATA] [-l LABELS]
	
	optional arguments:
	  -h, --help            show this help message and exit
	  -f {tune,train_eval}, --function {tune,train_eval}
	                        choose the function of the code
	  -d DATA, --data DATA  choose the folder where the data is
	  -l LABELS, --labels LABELS
	                        choose the folder where the labels are
	```
	
* An example usage is (takes about 5-10 minutes for the full output): 
	 ```
	 python code/main.py -f train_eval -d /path/to/data -l /path/to/labels
	 ```

* An example output where we can see our final classification scores as well as the most important features:

	  	Low SA02
		Most important feature, Oxygen saturation in timestep 2
		Low heartrate
		Most important feature, Respiratory rate in timestep 0
		Low respiration
		Most important feature, Heart rate in timestep 1
		Low Systemic Mean
		Most important feature, Heart rate in timestep 1
		High Heartrate
		Most important feature, Respiratory rate in timestep 0
		High respiration
		Most important feature, Respiratory rate in timestep 1
		High Systemic Mean
		Most important feature, Heart rate in timestep 2
		+--------------------+--------------------+--------------------+
		|        Task        |       AUROC        |       AURPC        |
		+--------------------+--------------------+--------------------+
		|      Low SA02      | 0.9912229148357453 | 0.991762160142757  |
		|   Low heartrate    | 0.9924621118928509 | 0.9925487250898325 |
		|  Low respiration   | 0.9896811148978384 | 0.9889808977160514 |
		| Low Systemic Mean  | 0.988212057593581  | 0.9906099666233024 |
		|   High Heartrate   | 0.9918989261768103 | 0.9891691887319317 |
		|  High respiration  | 0.9539927285772434 | 0.9551514355480142 |
		| High Systemic Mean | 0.9930555555555556 | 0.9934091623448059 |
		+--------------------+--------------------+--------------------+

* If you want you can also run the gridsearch to see why we chose these hyperparameters (but it will take 2-3 days on a single core):
``` python code/main.py -f tune -d /path/to/data -l /path/to/labels ```