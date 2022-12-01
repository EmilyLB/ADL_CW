# ADL

This is a repository for the Applied Deep Learning Coursework.

Link to Overleaf report: https://www.overleaf.com/2297789471yzcsbgcffjyw

Authors: Ambika Agarwal, Emily Lopez Burst

CHEAT SHEET
Copy file from local computer to BC4: scp FILE_NAME bc4-external:DESTINATION
Copy folder from local computer to BC4: scp -r FOLDER_NAME bc4-external:DESTINATION

Run code on bc4 for 30 seconds: sbatch --time 0-00:30 --mem 16GB --gres gpu:1

Module sometimes missing: module load languages/anaconda3/2021-3.8.8-cuda-11.1-pytorch

Tensorboard: 
In bc4:
    PORT=$((($UID-6025) % 65274))
    tensorboard --logdir logs --port $PORT --bind_all

In another terminal (not bc4):
    ssh -N -L 6006:localhost:port bc4-external
    
Remove folder with stuff in it: rm -rf <folder_name>
    


Next steps (22/11):
    1. Add in validation/accuracy code
    2. Add in optimisation/regularisation
    3. Edit print statements
    4. Try running on BC4
    5. Play around with the hyperparameters
    
  
Next steps (for 24/11):
    1. independently run code for 100 and 200 epochs
    2. start on tensorboard - loss curve, accuracy curve, confusion matrix
    3. Deep CNN.
    
Next steps (for 25/11):
    1. Ask TAs questions
        - Softmax
        - Imports and confusion matrix visualisation
        - Runtime
    2. Finish cross-val
    3. Work out next-steps for writing the report
    
 Next steps (for weekend)
  1. Run it without cross-validation and get accurarcy for training and test 
  
  2. Run it with cross-validation and get accuracy for training and test (for both, keep batch size=64 and do 100 epochs)
  
  3. Start reading papers for related work (2016 onwards) and also consider extension , let each other know what we are reading.
  
  Remaining steps for code:
    0. Merge branches
    
    1. Hyperparameter tuning (batch size, learning rate and scheduler) for ext2
    
    2. Different models of extension 2 (resnet18, resnet34, resnet50)
    
    3. Hyperparameter tuning for base_shallow (batch size)
    
    4. Comment our code (save this for after Friday)

