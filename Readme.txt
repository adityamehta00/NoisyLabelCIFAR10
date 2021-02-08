1.  The Training script is under "noisy_label_analysis.py"
    To use the training script use on terminal : python noisy_label_analysis.py --model_name model1 --alpha 0.1 --beta 0.6  --gpu-id 1 --epochs 120
    [NOTE other Hyperparameters are set to default. You can chage them by passing arguments in above command. Please refer the "noisy_label_analysis.py" to see the arguments]
  
2.  To test the trained model use "test.py" as:
    python test.py --model_name model1 --epoch 40
    
3.  The results are displayed on terminal. The accuracy and loss are automatically stored inside checkpoints/model_name/ as csv. 

4.  Report.pdf contains the required experiments and results. 
