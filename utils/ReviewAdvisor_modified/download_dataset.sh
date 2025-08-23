# revised, 2024/08

$ASAP_DATA_DIR="data/dataset/ASAP.zip"
perl scripts/gdown.pl https://drive.google.com/file/d/1nJdljy468roUcKLbVwWUhMs7teirah75/view?usp=sharing $ASAP_DATA_DIR
unzip $ASAP_DATA_DIR 
rm $ASAP_DATA_DIR