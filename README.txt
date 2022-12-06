Contents of the coursework.zip
-train_GTZAN_ext.py
-train_GTZAN_TL.py
-train.sh

To run Shallow CNN which is the base model from Schindler et al.
- Run train_GTZAN_ext.py as is submitted.
- Use the following parameters in train.sh
	- --time 0-00:30 --mem 24GB --gres gpu:1 (for 100 epochs)
	- --time 0-00:45 --mem 24GB --gres gpu:1 (for 200 epochs)
- We submit the code set at 100 epochs, but this can be changed to 200 on line 143

To run our first extension (cross validation) 
- Open train_GTZAN_ext.py
- In main, line number 177 uncomment this line (this runs cross-val)
- In main, line number 176 comment this line (this runs the shallow cnn model)
- Use the following parameters in train.sh
	- --time 0-01:45 --mem 24GB --gres gpu:1 (for 100 epochs)
	- --time 0-03:00 --mem 24GB --gres gpu:1 (for 200 epochs)
- We submit the code set at 100 epochs, but this can be changed to 200 on line 107

To run our final extension (transfer learning)
- Run train_GTZAN_TL.py
- Use the following parameters in train.sh
	- --time 0-00:45 --mem 24GB --gres gpu:1 
