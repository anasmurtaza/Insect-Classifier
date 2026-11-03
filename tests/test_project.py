import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image
from db import Database
from config import CLASSES
class DatabaseTests(unittest.TestCase):
    def test_security_ownership_and_images(self):
        with tempfile.TemporaryDirectory() as d:
            db=Database(Path(d)/'database.sqlite3')
            db.register('anas','strong-password');db.register('other','other-password')
            with self.assertRaises(ValueError):db.login('anas',"' OR 1=1 --")
            with self.assertRaises(ValueError):db.register('anas','different-password')
            a=db.login('anas','strong-password');b=db.login('other','other-password')
            with db.connect() as c:
                r=c.execute('SELECT * FROM users WHERE id=?',(a['id'],)).fetchone()
                self.assertNotEqual(r['password_hash'],'strong-password');self.assertTrue(r['salt'])
            image=Path(d)/'test.png';Image.new('RGB',(32,32),(120,70,40)).save(image)
            row=db.save(a['id'],image,'bee',.75,'beetle')
            self.assertEqual(len(db.list_records(a['id'])),1);self.assertEqual(db.list_records(b['id']),[])
            saved=Path(db.list_records(a['id'])[0]['image_path']);self.assertTrue(saved.exists())
            with self.assertRaises(ValueError):db.delete(b['id'],row)
            self.assertTrue(saved.exists());db.delete(a['id'],row);self.assertFalse(saved.exists())
    def test_network_and_checkpoint(self):
        import torch
        from model import InsectCNN,load_model
        model=InsectCNN();x=torch.rand(2,3,32,32);y=model(x)
        self.assertEqual(tuple(y.shape),(2,5))
        loss=torch.nn.CrossEntropyLoss()(y,torch.tensor([0,4]));loss.backward()
        allowed=(torch.nn.Conv2d,torch.nn.ReLU,torch.nn.MaxPool2d,torch.nn.Flatten,torch.nn.Linear,torch.nn.Dropout)
        for layer in model.network:self.assertIsInstance(layer,allowed)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model.pt';torch.save({'state_dict':model.state_dict(),'classes':CLASSES},p)
            restored=load_model(p);self.assertFalse(restored.training)
            from predict import Predictor
            image=Path(d)/'x.png';Image.new('RGB',(70,50)).save(image)
            probs=Predictor(p).predict(image)
            self.assertEqual(len(probs),5);self.assertAlmostEqual(sum(p for _,p in probs),1,places=5)
if __name__=='__main__':unittest.main()
