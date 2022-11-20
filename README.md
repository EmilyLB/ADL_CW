# ADL

This is a repository for the Applied Deep Learning Coursework.

Link to Overleaf report: https://www.overleaf.com/2297789471yzcsbgcffjyw

Authors: Ambika Agarwal, Emily Lopez Burst

CHEAT SHEET
Copy file from local computer to BC4: scp FILE_NAME bc4-external:DESTINATION
Copy folder from local computer to BC4: scp -r FOLDER_NAME bc4-external:DESTINATION

Run code on bc4 for 30 seconds: sbatch --time 0-00:30 --mem 16GB --gres gpu:1

Module sometimes missing: module load languages/anaconda3/2021-3.8.8-cuda-11.1-pytorch


Next steps (20/11):
    1. Add in validation/accuracy code
    2. Add in optimisation/regularisation
    3. Edit print statements
    4. Try running on BC4
    5. Play around with the hyperparameters