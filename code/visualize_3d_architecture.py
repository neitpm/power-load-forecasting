# -*- coding: utf-8 -*-
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
OUT=Path(__file__).parents[1]/'docs'/'figures'; OUT.mkdir(parents=True,exist_ok=True)
def make(fn,title,items):
 fig,ax=plt.subplots(figsize=(16,9),dpi=300); fig.patch.set_facecolor('#0b1120'); ax.set_facecolor('#0b1120'); ax.set_xlim(0,16); ax.set_ylim(0,9); ax.axis('off'); ax.text(8,8.3,title,ha='center',color='white',fontsize=17,weight='bold')
 for i,(a,b,c,col) in enumerate(items):
  x=.5+i*3.55;y=3.2;w=2.7;h=2.4;dx=.28;dy=.22
  ax.add_patch(Polygon([(x+w,y),(x+w+dx,y+dy),(x+w+dx,y+h+dy),(x+w,y+h)],fc=col,ec='#020617')); ax.add_patch(Polygon([(x,y+h),(x+dx,y+h+dy),(x+w+dx,y+h+dy),(x+w,y+h)],fc='#a5b4fc',ec='#020617')); ax.add_patch(Polygon([(x,y),(x+w,y),(x+w,y+h),(x,y+h)],fc=col,ec='#020617'))
  ax.text(x+w/2,y+h*.62,a,ha='center',color='white',weight='bold',fontsize=9); ax.text(x+w/2,y+h*.38,b,ha='center',color='white',fontsize=7); ax.text(x+w/2,y+h*.14,c,ha='center',color='#fef08a',fontsize=7,family='monospace')
  if i<3: ax.annotate('',xy=(x+3.35,4.4),xytext=(x+2.75,4.4),arrowprops=dict(arrowstyle='-|>',color='#38bdf8',lw=2))
 fig.savefig(OUT/fn,dpi=300,bbox_inches='tight'); plt.close(fig)
if __name__=='__main__':
 d='#0284c7';p='#475569';m='#7c3aed';e='#059669';o='#db2777'
 make('1_data_pipeline_3d.png','Industrial Power Load Forecasting: Data Pipeline',[('Raw Metering','3-phase measurements','2020–2025',d),('Time Alignment','30-minute records','P,Q,U,I,cosφ',p),('Quality Control','Gaps + outliers','16 test segments',p),('Model Table','Chronological split','Train / Val / Test',e)])
 make('2_preprocessing_contract_3d.png','Preprocessing & Leakage Control',[('Train Fit','Imputer + scaler','fit: train only',d),('Validation','Transform only','early stopping',p),('Test','Out-of-sample','14,385 records',e),('Bootstrap','Contiguous blocks','16 segments',o)])
 make('3_model_benchmark_3d.png','Model Benchmark Matrix',[('Naive','Persistence / seasonal','strong baselines',d),('Classical ML','Ridge / XGBoost','tree ensembles',p),('Residual KAN','KAN + residual path','5 seeds',m),('Ensemble','Uniform averaging','MAE 519.78 kW',e)])
 make('4_kan_ensemble_3d.png','Residual KAN Multi-Seed Ensemble',[('Windowed Features','Lagged load + features','fixed input',d),('KAN Model','Learnable functions','Residual path',m),('Five Seeds','42 / 101 / 2024','7 / 999',m),('Average','1/5 predictions','zero test tuning',e)])
 make('5_evaluation_3d.png','Evaluation & Error Analysis',[('MAE','Absolute error','519.78 kW',d),('RMSE','Large-error penalty','1019.95 kW',m),('R²','Explained variance','0.9050',e),('sMAPE','Naive remains best','13.35%',o)])


