from joblib import dump, load
import json
import sys
home_encoded=sys.argv[1];
away_encoded=sys.argv[2];
HTHG=sys.argv[3];
HTAG=sys.argv[4];
HS=sys.argv[5];
AS=sys.argv[6];
HST=sys.argv[7];
AST=sys.argv[8];
HR=sys.argv[9];
AR=sys.argv[10];
lr_classifier = load('C:\\Users\\alifa\\Desktop\\proje\\exportedModels\\lr_classifier.model')
data=lr_classifier.predict_proba([[home_encoded, away_encoded, HTHG, HTAG, HS,AS, HST, AST, HR, AR]])
dataToString=str(data[0])
dataToString=dataToString.replace('[','')
dataToString=dataToString.replace(']','')
x = dataToString.split(" ")
myDic ={"AwayWin":x[0],"draw":x[1],"HomeWin":x[2]};
print(json.dumps(myDic))

