echo '----------------Vanilla Model-----------------'
python3 calculateFscore.py data/train-labeled.txt data/dev-text.txt data/dev-key.txt vanillamodel.txt
echo '----------------Average Model-----------------'
python3 calculateFscore.py data/train-labeled.txt data/dev-text.txt data/dev-key.txt averagedmodel.txt
