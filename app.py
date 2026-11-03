"""Desktop login/prediction/history app. GUI never retrains at startup."""
import tkinter as tk
from tkinter import ttk,filedialog,messagebox
from pathlib import Path
from PIL import Image,ImageOps,ImageTk
from db import Database
from config import CLASSES,OUTPUT_DIR
class App:
    def __init__(self,root):
        self.root=root;self.db=Database();self.user=None;self.predictor=None
        root.title('Insect Classifier | Muhammad Anas Murtaza | Project 31');root.geometry('950x690');root.minsize(820,600)
        style=ttk.Style();style.theme_use('clam')
        import tkinter.font as tkfont
        for font_name in ['TkDefaultFont','TkTextFont','TkMenuFont']:
            tkfont.nametofont(font_name).configure(family='Helvetica',size=11)
        self.font_family=tkfont.nametofont('TkDefaultFont').actual('family')
        style.configure('TButton',padding=8);style.configure('Title.TLabel',font=(self.font_family,23,'bold'),foreground='#203e7b')
        self.login_screen()
    def clear(self):
        for w in self.root.winfo_children():w.destroy()
    def login_screen(self):
        self.clear();self.user=None;self.predictor=None
        panel=ttk.Frame(self.root,padding=45);panel.pack(expand=True)
        ttk.Label(panel,text='INSECT CLASSIFIER',style='Title.TLabel').pack(pady=12)
        ttk.Label(panel,text='5 classes · CIFAR-100 · CNN from scratch').pack(pady=5)
        ttk.Label(panel,text='Username').pack(anchor='w',pady=(22,3));self.username=ttk.Entry(panel,width=35);self.username.pack()
        ttk.Label(panel,text='Password').pack(anchor='w',pady=(12,3));self.password=ttk.Entry(panel,width=35,show='*');self.password.pack()
        ttk.Button(panel,text='Login',command=self.login).pack(fill='x',pady=(22,5))
        ttk.Button(panel,text='Register new account',command=self.register).pack(fill='x')
        ttk.Label(panel,text='Muhammad Anas Murtaza | DF2-AI-2026-031').pack(pady=25)
    def register(self):
        try:self.db.register(self.username.get(),self.password.get());messagebox.showinfo('Registered','Account created. Now log in.')
        except Exception as e:messagebox.showerror('Register',str(e))
    def login(self):
        try:self.user=self.db.login(self.username.get(),self.password.get());self.workspace()
        except Exception as e:messagebox.showerror('Login',str(e))
    def workspace(self):
        self.clear();self.path=None;self.result=None
        header=ttk.Frame(self.root,padding=12);header.pack(fill='x')
        ttk.Label(header,text='Insect Classifier',style='Title.TLabel').pack(side='left')
        ttk.Button(header,text='Logout',command=self.login_screen).pack(side='right');ttk.Label(header,text=self.user['username']+'  ').pack(side='right')
        tabs=ttk.Notebook(self.root);tabs.pack(fill='both',expand=True,padx=15,pady=10)
        predict_tab=ttk.Frame(tabs,padding=15);history_tab=ttk.Frame(tabs,padding=15)
        tabs.add(predict_tab,text='Predict');tabs.add(history_tab,text='My history')
        left=ttk.Frame(predict_tab);left.pack(side='left',fill='both',expand=True)
        right=ttk.Frame(predict_tab,padding=20);right.pack(side='right',fill='both',expand=True)
        self.preview=ttk.Label(left,text='Open an insect image',anchor='center');self.preview.pack(fill='both',expand=True)
        ttk.Button(left,text='Open image and predict',command=self.open_image).pack(fill='x',pady=12)
        self.prediction=ttk.Label(right,text='Model predictions appear here',font=(self.font_family,15,'bold'),wraplength=340);self.prediction.pack(anchor='w',pady=18)
        self.top3=ttk.Label(right,text='',font=(self.font_family,12),justify='left');self.top3.pack(anchor='w',pady=15)
        ttk.Label(right,text='True label (optional)').pack(anchor='w',pady=(15,5));self.true=ttk.Combobox(right,values=['']+CLASSES,state='readonly');self.true.set('');self.true.pack(anchor='w')
        self.save_button=ttk.Button(right,text='Save to my history',command=self.save,state='disabled');self.save_button.pack(anchor='w',pady=16)
        ttk.Label(right,text='Only these 5 insects are supported.\nConfidence is a model estimate, not certainty.\nLarge photos are resized to 32 × 32.',wraplength=330).pack(anchor='w',pady=8)
        ttk.Button(right,text='View test confusion matrix',command=self.show_matrix).pack(anchor='w',pady=10)
        self.tree=ttk.Treeview(history_tab,columns=('label','confidence','true','date'),show='headings',height=9,selectmode='browse')
        for col in self.tree['columns']:self.tree.heading(col,text=col.title());self.tree.column(col,width=150)
        self.tree.pack(fill='x');self.tree.bind('<<TreeviewSelect>>',self.history_preview)
        self.old_preview=ttk.Label(history_tab,text='Select a saved record');self.old_preview.pack(pady=10)
        buttons=ttk.Frame(history_tab);buttons.pack(fill='x')
        for label,fn in [('Refresh',self.refresh),('Delete selected',self.delete),('Export my CSV',self.export)]:ttk.Button(buttons,text=label,command=fn).pack(side='left',padx=4)
        self.refresh()
    def open_image(self):
        path=filedialog.askopenfilename(filetypes=[('Images','*.png *.jpg *.jpeg *.bmp *.webp'),('All files','*.*')])
        if not path:return
        self.result=None;self.path=None;self.save_button.config(state='disabled');self.prediction.config(text='Loading...');self.top3.config(text='');self.root.update_idletasks()
        try:
            if self.predictor is None:
                from predict import Predictor
                self.predictor=Predictor()
            results=self.predictor.predict(path)
            with Image.open(path) as im:
                img=ImageOps.exif_transpose(im).convert('RGB');img=ImageOps.contain(img,(320,320),method=Image.Resampling.NEAREST if img.width<=64 else Image.Resampling.LANCZOS);self.photo=ImageTk.PhotoImage(img)
            self.preview.config(image=self.photo,text='');self.path=path;self.result=results
            self.prediction.config(text=f'{results[0][0].title()} — {results[0][1]:.1%}')
            self.top3.config(text='TOP 3 RESULTS\n\n'+'\n'.join(f'{i+1}. {name.title():15} {p:.1%}' for i,(name,p) in enumerate(results[:3])))
            self.save_button.config(state='normal')
        except Exception as e:self.prediction.config(text='Prediction unavailable');messagebox.showerror('Prediction',str(e))
    def save(self):
        if not self.path or not self.result:return
        try:
            self.db.save(self.user['id'],self.path,*self.result[0],self.true.get());self.refresh();self.save_button.config(state='disabled');messagebox.showinfo('Saved','Image and prediction saved to your history.')
        except Exception as e:messagebox.showerror('Save',str(e))
    def refresh(self):
        self.tree.delete(*self.tree.get_children());self.records={r['id']:r for r in self.db.list_records(self.user['id'])}
        for r in self.records.values():self.tree.insert('', 'end',iid=str(r['id']),values=(r['predicted_label'],f"{r['confidence']:.1%}",r['true_label'] or '',r['created_at'][:19]))
        self.old_preview.config(image='',text='Select a saved record')
    def history_preview(self,event=None):
        selection=self.tree.selection()
        if not selection:return
        try:
            with Image.open(self.records[int(selection[0])]['image_path']) as im:
                preview=ImageOps.contain(im,(190,160),method=Image.Resampling.NEAREST if im.width<=64 else Image.Resampling.LANCZOS);self.history_photo=ImageTk.PhotoImage(preview)
            self.old_preview.config(image=self.history_photo,text='')
        except Exception as e:messagebox.showerror('Preview',str(e))
    def delete(self):
        selected=self.tree.selection()
        if selected and messagebox.askyesno('Delete','Delete this saved image and record?'):
            try:self.db.delete(self.user['id'],int(selected[0]));self.refresh()
            except Exception as e:messagebox.showerror('Delete',str(e))
    def export(self):
        path=filedialog.asksaveasfilename(defaultextension='.csv',initialfile='my_insect_history.csv')
        if path:
            try:self.db.export(self.user['id'],path);messagebox.showinfo('Export','History exported.')
            except Exception as e:messagebox.showerror('Export',str(e))
    def show_matrix(self):
        path=OUTPUT_DIR/'confusion_matrix.png'
        if not path.exists():messagebox.showinfo('Evaluation','Run train.py first to generate the confusion matrix.');return
        win=tk.Toplevel(self.root);win.title('Official test confusion matrix')
        with Image.open(path) as im:im.thumbnail((700,550));photo=ImageTk.PhotoImage(im.copy())
        label=ttk.Label(win,image=photo);label.image=photo;label.pack(padx=10,pady=10)
if __name__=='__main__':
    root=tk.Tk();App(root);root.mainloop()
