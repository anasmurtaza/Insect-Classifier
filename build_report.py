"""Six-page report populated only from measured training outputs and real screenshots."""
import json
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor,white
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Paragraph,Table,TableStyle
from reportlab.lib.styles import ParagraphStyle
from config import ROOT,OUTPUT_DIR,CLASSES
pdfmetrics.registerFont(TTFont('ProjectSans',str(ROOT/'assets'/'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('ProjectSansBold',str(ROOT/'assets'/'DejaVuSans-Bold.ttf')))
pdfmetrics.registerFontFamily('ProjectSans',normal='ProjectSans',bold='ProjectSansBold',italic='ProjectSans',boldItalic='ProjectSansBold')
W,H=595.28,841.89
BLUE=HexColor('#22477a');MUTED=HexColor('#5c6b80')
style=ParagraphStyle('body',fontName='ProjectSans',fontSize=10,leading=15,textColor=HexColor('#26364a'))
def build():
    metrics=json.loads((OUTPUT_DIR/'metrics.json').read_text()) if (OUTPUT_DIR/'metrics.json').exists() else None
    c=canvas.Canvas(str(ROOT/'report.pdf'),pagesize=(W,H));c.setTitle('Project 31 - Insect Classifier - Muhammad Anas Murtaza')
    def page(number,title):
        c.setFillColor(BLUE);c.rect(0,H-110,W,110,fill=1,stroke=0);c.setFillColor(white)
        c.setFont('ProjectSans',9);c.drawString(38,H-32,'HITEC UNIVERSITY, TAXILA  /  CNN PROJECT 31')
        c.setFont('ProjectSansBold',21);c.drawString(38,H-70,title)
        c.setFillColor(MUTED);c.setFont('ProjectSans',8);c.drawString(38,26,'Muhammad Anas Murtaza | DF2-AI-2026-031');c.drawRightString(W-38,26,f'{number} / 6')
    def para(text,y,size=10):
        st=ParagraphStyle('p',parent=style,fontSize=size,leading=size*1.5)
        p=Paragraph(text,st);_,h=p.wrap(W-76,700);p.drawOn(c,38,y-h);return y-h-12
    def heading(text,y):
        c.setFillColor(BLUE);c.setFont('ProjectSansBold',12);c.drawString(38,y-13,text);return y-27
    def image(name,y,max_height=210,width=None):
        path=OUTPUT_DIR/name;width=width or W-76
        if not path.exists():return para('Evidence unavailable: '+name+'. Run training or capture the real application screen before final submission.',y)
        from PIL import Image
        with Image.open(path) as im:iw,ih=im.size
        scale=min(width/iw,max_height/ih);dw,dh=iw*scale,ih*scale
        c.drawImage(str(path),38+(W-76-dw)/2,y-dh,width=dw,height=dh,mask='auto');return y-dh-12
    def table(rows,y,widths):
        t=Table(rows,colWidths=widths);t.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'ProjectSans'),('BACKGROUND',(0,0),(-1,0),BLUE),('TEXTCOLOR',(0,0),(-1,0),white),('FONTNAME',(0,0),(-1,0),'ProjectSansBold'),('FONTSIZE',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),8),('GRID',(0,0),(-1,-1),.4,HexColor('#d8dfeb')),('VALIGN',(0,0),(-1,-1),'TOP')]))
        _,h=t.wrap(W-76,H);t.drawOn(c,38,y-h);return y-h-16
    page(1,'Insect Classifier')
    y=H-140
    y=para('<b>Student:</b> Muhammad Anas Murtaza<br/><b>Registration:</b> DF2-AI-2026-031<br/><b>Task:</b> Five-class object recognition from CIFAR-100; 32 x 32 RGB input.',y)
    y=heading('Problem and dataset',y)
    y=para('The system recognises bee, beetle, butterfly, caterpillar and cockroach. A CNN is trained from scratch using only the permitted basic layers. A desktop GUI provides account registration, login, prediction, saved-image history and logout.',y)
    y=para('CIFAR-100 has 500 official training and 100 official test images per fine class. Filtering by class name gives 2,500 training-pool images and 500 test images. A stratified 80/20 split of the training pool gives 2,000 fitting images and 500 validation images. Labels are remapped in the listed order to 0 through 4.',y)
    y=heading('One official test example per class',y);y=image('samples.png',y,145)
    y=heading('Preprocessing and augmentation',y)
    y=para('RGB conversion, resize to 32 x 32, and ToTensor scale pixels to [0,1]. Training only uses horizontal flips, rotations up to 10 degrees and translations up to 8%. Validation, test and GUI inference use deterministic preprocessing.',y)
    y=para('Source: Krizhevsky (2009), Learning Multiple Layers of Features from Tiny Images.<br/>www.cs.toronto.edu/~kriz/cifar.html<br/>Dataset files are downloaded automatically and excluded from the submitted ZIP.',y,9)
    c.showPage();page(2,'CNN architecture')
    y=H-140
    rows=[['Layer / operation','Parameters','Output (excluding batch)'],['Input','RGB, normalised [0,1]','3 x 32 x 32'],['Conv2d + ReLU','3 -> 32, kernel 3, padding 1','32 x 32 x 32'],['MaxPool2d','kernel 2, stride 2','32 x 16 x 16'],['Conv2d + ReLU','32 -> 64, kernel 3, padding 1','64 x 16 x 16'],['MaxPool2d','kernel 2, stride 2','64 x 8 x 8'],['Conv2d + ReLU','64 -> 128, kernel 3, padding 1','128 x 8 x 8'],['MaxPool2d','kernel 2, stride 2','128 x 4 x 4'],['Flatten','128 * 4 * 4','2048'],['Linear + ReLU','2048 -> 128','128'],['Dropout','p = 0.4 (training only)','128'],['Output Linear','128 -> 5 raw logits','5']]
    y=table(rows,y,[145,180,194])
    y=heading('Why these layers?',y)
    y=para('Convolution filters learn local edge, texture and shape patterns. ReLU introduces nonlinearity. Pooling reduces spatial size. The hidden Linear layer combines learned features, and Dropout regularises the classifier. Three convolution blocks exceed the required minimum of two.',y)
    y=para('No pretrained model, transfer learning, BatchNorm or adaptive pooling is used. CrossEntropyLoss receives raw logits; softmax is applied only for inference confidence. The network has 356,165 trainable parameters.',y)
    c.showPage();page(3,'Training and learning curves')
    y=H-140
    y=para('Adam optimiser; learning rate 0.001; weight decay 0.0001; batch size 64; CrossEntropyLoss; random seed 42. The checkpoint is selected by validation accuracy. Train/validation/test metrics are printed each epoch, but test results are not used to choose the checkpoint or tune settings.',y)
    if metrics:y=para(f"Measured run: {metrics['epochs']} epochs on {metrics['device']}; selected epoch {metrics['checkpoint_epoch']}; elapsed {metrics['elapsed_seconds']:.1f} seconds. Split sizes: {metrics['train_images']} train / {metrics['validation_images']} validation / {metrics['test_images']} test.",y)
    else:y=para('<b>No completed measured training run is available.</b>',y)
    y=image('curves.png',y,240)
    y=heading('Interpreting the curves',y)
    y=para('Training loss measures fit to augmented training images. Validation loss and accuracy measure generalisation on held-out training-pool images. A widening train/validation gap or rising validation loss while training improves is evidence consistent with overfitting. Test curves are monitoring evidence only.',y)
    y=para('The saved state_dict contains learned weights; class order, input size and normalisation metadata accompany it. The GUI loads the checkpoint in evaluation mode, so Dropout is disabled. It never retrains during startup.',y)
    c.showPage();page(4,'Evaluation and error analysis')
    y=H-137
    if metrics:
        y=para(f"<b>Official held-out test accuracy: {metrics['test_accuracy']:.2%}</b> on {metrics['test_images']} images. Per-class values are recalls (correct examples divided by all true examples in that class).",y)
        rows=[['Class','Per-class accuracy']]+[[name,f"{metrics['per_class_accuracy'][name]:.2%}"] for name in CLASSES]
        y=table(rows,y,[260,259])
    if metrics:
        per_class=metrics['per_class_accuracy'];best_class=max(per_class,key=per_class.get);worst_class=min(per_class,key=per_class.get)
        cm=metrics['confusion_matrix'];i=CLASSES.index(worst_class);j=max((k for k in range(5) if k!=i),key=lambda k:cm[i][k])
        y=para(f'{best_class.title()} performs best ({per_class[best_class]:.0%}); {worst_class} is weakest ({per_class[worst_class]:.0%}). Of {sum(cm[i])} true {worst_class} images, {cm[i][j]} are predicted as {CLASSES[j]}. Shared visual cues are one possible explanation, not an established cause.',y,8)
    y=image('confusion_matrix.png',y,170)
    y=image('wrong_predictions.png',y,110)
    errors=json.loads((OUTPUT_DIR/'wrong_predictions.json').read_text()) if (OUTPUT_DIR/'wrong_predictions.json').exists() else []
    for error in errors[:3]:
        y=para(f"<b>{error['true']} mistaken for {error['predicted']}:</b> Possible small-image ambiguity, shared colour/texture or background cues. This visual interpretation is a hypothesis, not a proven causal explanation.",y,8)
    if len(errors)<3:y=para(f'Only {len(errors)} saved wrong examples are available. No additional mistakes are fabricated.',y,8)
    c.showPage();page(5,'Database design and security')
    y=H-140
    y=heading('Relational schema: one user to many saved images',y)
    y=table([['users','image_records'],['id INTEGER PRIMARY KEY','id INTEGER PRIMARY KEY'],['username TEXT UNIQUE NOT NULL','user_id INTEGER FK -> users.id'],['salt TEXT NOT NULL','image_path TEXT NOT NULL'],['password_hash TEXT NOT NULL','predicted_label TEXT NOT NULL'],['created_at TEXT NOT NULL','confidence REAL NOT NULL'],['','true_label TEXT NULL; created_at TEXT']],y,[259,260])
    y=para('A new SQLite database is created automatically. Passwords are salted with 16 cryptographically random bytes and hashed using PBKDF2-HMAC-SHA256 with 600,000 iterations. Login recomputes the hash and uses constant-time comparison. Passwords are never stored as plain text.',y)
    y=para('All SQL input uses bound parameters. History and deletion queries include the logged-in user ID. Foreign keys are enabled. The app copies each saved image as a uniquely named RGB PNG into app_data/records; deleting a record removes the app copy, not the original source file.',y)
    y=heading('Workflow and tested behaviour',y)
    y=para('Register / login -> open image -> preview and top-3 prediction -> optional true label -> Save -> user history -> preview/delete/CSV export -> logout. Missing models, invalid images and wrong credentials produce clear dialogs.',y)
    y=para('Automated checks cover salted authentication, duplicate usernames, SQL-like password input, cross-user record isolation, ownership-protected deletion, copied-image cleanup, permitted layer types, tensor shape, backward propagation, checkpoint loading and probability normalisation.',y)
    y=heading('Limits',y)
    y=para('The model always chooses one of the five classes, including for out-of-scope photos. Confidence is uncalibrated. Resizing to 32 x 32 loses detail, and personal photos may differ from CIFAR. SQLite is local and not encrypted; filesystem permissions still matter.',y)
    c.showPage();page(6,'Desktop application and conclusion')
    y=H-135
    for name,title in [('gui_login.png','1. Login / registration'),('gui_predict.png','2. Prediction, confidence and top-3'),('gui_history.png','3. Saved history with preview')]:
        y=heading(title,y);y=image(name,y,145)
    y=para('The application connects a basic CNN to secure user accounts and per-user image storage. Future work could improve validation-based tuning, calibration and performance on realistic insect photos. Review measured errors and be ready to explain every module during the individual viva.',y,8)
    c.save();print(ROOT/'report.pdf')
if __name__=='__main__':build()
