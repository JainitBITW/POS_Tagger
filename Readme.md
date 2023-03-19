# Readme

Dataset used : [UD_English-Atis](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-4923) 

## File structure
* 2021114003_NLP_ASS2
  * **datasets** contains English dataset  
  * **Report.pdf**  
  * **ModelTrainer.py** -> Trains the model of specified but requires execution file 
  * **TrainModel.py** -> Execution file to save and make new model
  * **model1.pt** -> pre-trained model and saving dicts of word to index and tags to index 
  * **pos_tagger.py** -> for testing , tagging a input sentence
  * **DataMaker.py** -> Makes Dataloader for datasets using TOrch's DataLoader


## To run the script 
### To Train the model  
```bash
python3 TrainModel.py
```
This will also ask for various hyperparameters. You can play with them.
### To run pos tagger 
```bash
python3 pos_tagger.py
```
after running this command, there will be a prompt which will ask for sentence for which you want POS tags. 
