# Global and Local Interpretable CNN (GL-ICNN) for AD diagnosis and prediction

preprint:  


### Model building
The model is built based on Pytorch framework.   
GL_ICNN.py: code to build the GL-ICNN model, and used in the model training. Including loops to generate CNN structures of different pathways, and a output block consisting of EBM.   

### Model training 
The GL-ICNN is train in two stages, and CNN part and EBM part train alternatively on the second stage.     
model_training.py: the code to train the model. And save the best model, learning curves, and output features.   


### Model testing
The save model in pth format is too large that cannot put into Gitlab. So the output features from the CNN part of the saved GL-ICNN, including the data and label of training and testing sets, were save to the data folder.   
model_testing.py: training the EBM (based on the saved best GL-ICNN)，and testing on the split/external testing set. The code can provide performance metrics, group-level feature importance, and individual feature importance. The figures will be saved to the plot folder, and there are example figures in the plot folder already.   



















