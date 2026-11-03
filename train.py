import argparse,json,time,random,csv
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import CLASSES,MODEL_PATH,OUTPUT_DIR
from model import InsectCNN,load_model
from data import datasets

def epoch(model,loader,loss_fn,device,optimizer=None):
    model.train(optimizer is not None);total=correct=0;loss_sum=0.
    for images,labels in loader:
        images,labels=images.to(device),labels.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            logits=model(images);loss=loss_fn(logits,labels)
            if optimizer is not None:
                optimizer.zero_grad();loss.backward();optimizer.step()
        total+=len(labels);loss_sum+=loss.item()*len(labels);correct+=(logits.argmax(1)==labels).sum().item()
    return loss_sum/total,correct/total

def train(epochs=30,batch_size=64,lr=.001,data_root=None,device='auto',seed=42):
    if epochs<1 or batch_size<1:raise ValueError('epochs and batch_size must be positive')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    device=('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device
    OUTPUT_DIR.mkdir(exist_ok=True);MODEL_PATH.parent.mkdir(exist_ok=True)
    fit,val,test=datasets(data_root,seed)
    gen=torch.Generator().manual_seed(seed)
    train_loader=DataLoader(fit,batch_size=batch_size,shuffle=True,generator=gen,num_workers=0)
    val_loader=DataLoader(val,batch_size=batch_size,num_workers=0)
    test_loader=DataLoader(test,batch_size=batch_size,num_workers=0)
    model=InsectCNN().to(device);optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-4)
    loss_fn=nn.CrossEntropyLoss();best=-1.;history=[];start=time.time()
    print(f'Device {device}; train {len(fit)}, validation {len(val)}, test {len(test)}',flush=True)
    for e in range(1,epochs+1):
        tl,ta=epoch(model,train_loader,loss_fn,device,optimizer)
        vl,va=epoch(model,val_loader,loss_fn,device)
        # Assignment asks for train/test logging. Test is monitoring only; never used for checkpoint selection.
        test_loss,test_acc=epoch(model,test_loader,loss_fn,device)
        row=dict(epoch=e,train_loss=tl,train_accuracy=ta,val_loss=vl,val_accuracy=va,test_loss=test_loss,test_accuracy=test_acc)
        history.append(row)
        print(f'Epoch {e:02d}/{epochs} train {tl:.4f}/{ta:.2%} val {vl:.4f}/{va:.2%} test {test_loss:.4f}/{test_acc:.2%}',flush=True)
        if va>best:
            best=va;torch.save({'state_dict':model.state_dict(),'classes':CLASSES,'epoch':e,'input_size':[3,32,32],'normalization':'RGB [0,1]'},MODEL_PATH)
        with open(OUTPUT_DIR/'history.csv','w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(history)
    model=load_model(MODEL_PATH,device)
    cm=np.zeros((5,5),dtype=int);wrong=[]
    model.eval()
    with torch.inference_mode():
        for images,labels in test_loader:
            probs=torch.softmax(model(images.to(device)),1).cpu();preds=probs.argmax(1)
            for image,true,pred,prob in zip(images,labels,preds,probs):
                t,p=int(true),int(pred);cm[t,p]+=1
                if t!=p and len(wrong)<3:wrong.append((image,t,p,float(prob[p])))
    metrics={'test_accuracy':float(cm.trace()/cm.sum()),'classes':CLASSES,'confusion_matrix':cm.tolist(),
             'per_class_accuracy':{name:float(cm[i,i]/cm[i].sum()) for i,name in enumerate(CLASSES)},
             'train_images':len(fit),'validation_images':len(val),'test_images':len(test),
             'epochs':epochs,'seed':seed,'elapsed_seconds':round(time.time()-start,2),'device':device,
             'torch_version':str(torch.__version__),'numpy_version':str(np.__version__),
             'checkpoint_epoch':int(torch.load(MODEL_PATH,map_location='cpu',weights_only=True)['epoch'])}
    (OUTPUT_DIR/'metrics.json').write_text(json.dumps(metrics,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    for prefix,label in [('train','Train'),('val','Validation'),('test','Test (monitor only)')]:
        axes[0].plot([r['epoch'] for r in history],[r[prefix+'_loss'] for r in history],label=label)
        axes[1].plot([r['epoch'] for r in history],[r[prefix+'_accuracy'] for r in history],label=label)
    for ax,title in zip(axes,['Cross-entropy loss','Accuracy']):ax.set_title(title);ax.set_xlabel('Epoch');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUTPUT_DIR/'curves.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5));im=ax.imshow(cm,cmap='Blues');fig.colorbar(im,ax=ax)
    ax.set(xticks=range(5),yticks=range(5),xticklabels=CLASSES,yticklabels=CLASSES,xlabel='Predicted',ylabel='True',title='Official test confusion matrix')
    plt.setp(ax.get_xticklabels(),rotation=25,ha='right')
    for i in range(5):
        for j in range(5):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
    fig.tight_layout();fig.savefig(OUTPUT_DIR/'confusion_matrix.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,5,figsize=(10,2.4))
    for i,ax in enumerate(axes):
        idx=next(j for j in range(len(test)) if test[j][1]==i);image,_=test[idx]
        ax.imshow(image.permute(1,2,0));ax.set_title(CLASSES[i]);ax.axis('off')
    fig.tight_layout();fig.savefig(OUTPUT_DIR/'samples.png',dpi=160);plt.close(fig)
    explanations=[]
    if wrong:
        fig,axes=plt.subplots(1,len(wrong),figsize=(3.2*len(wrong),3),squeeze=False)
        for k,(image,t,p,confidence) in enumerate(wrong):
            axes[0,k].imshow(image.permute(1,2,0));axes[0,k].axis('off');axes[0,k].set_title(f'True: {CLASSES[t]}\nPred: {CLASSES[p]} ({confidence:.1%})',fontsize=9)
            explanations.append({'true':CLASSES[t],'predicted':CLASSES[p],'confidence':confidence,'explanation':'Possible causes: 32x32 resolution, similar shape/colour or background. This is a hypothesis, not a proven model explanation.'})
        fig.tight_layout();fig.savefig(OUTPUT_DIR/'wrong_predictions.png',dpi=150);plt.close(fig)
    (OUTPUT_DIR/'wrong_predictions.json').write_text(json.dumps(explanations,indent=2))
    print(json.dumps(metrics,indent=2));return metrics

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=30);p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--lr',type=float,default=.001);p.add_argument('--data-root');p.add_argument('--device',default='auto',choices=['auto','cpu','cuda']);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();train(a.epochs,a.batch_size,a.lr,a.data_root,a.device,a.seed)
