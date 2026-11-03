# Project samjhein — Muhammad Anas Murtaza

## Ye project kya karta hai?
Aap insect ki image select karte hain. CNN usay bee, beetle, butterfly, caterpillar ya cockroach mein classify karta hai. App image ka preview, predicted name, confidence aur top-3 classes dikhati hai. Login ke baad aap image save karke apni history dekh sakte hain.

## 1. Dataset aur labels
CIFAR-100 mein 100 classes hain. Hum sirf assigned 5 classes lete hain. Original IDs ko names se find karke naye labels banate hain: bee=0, beetle=1, butterfly=2, caterpillar=3, cockroach=4. Is se model ke 5 output neurons sahi classes ke saath match rehte hain.

Har class ki official training images 500 aur test images 100 hain. Total selected training 2500 aur test 500. Training ke 2500 mein se har class ki 100 validation mein aur 400 actual training mein jati hain: 2000 train, 500 validation, 500 official test.

## 2. Preprocessing
Image RGB mein convert aur 32x32 resize hoti hai. ToTensor 0..255 pixels ko 0..1 floats mein convert karta hai; shape 3x32x32 hoti hai. Sirf training images par slight rotation/shift aur horizontal flip lagate hain, taake model sirf ek pose ya location yaad na kare. Validation/test/prediction par random augmentation nahi hoti.

## 3. CNN layers
- Conv2d 3 -> 32: 3x3 filters image ki local edges/textures seekhte hain. Padding size preserve karta hai.
- ReLU: negative values zero; non-linear patterns seekhne mein madad.
- MaxPool2d: width/height half; 32x32 -> 16x16.
- Second block: 32 -> 64 channels; pool ke baad 64x8x8.
- Third block: 64 -> 128 channels; pool ke baad 128x4x4.
- Flatten: 128*4*4 = 2048 numbers ki vector.
- Linear 2048 -> 128: learned features ko combine karta hai.
- ReLU + Dropout .4: training mein randomly 40% hidden activations drop; overfitting reduce karne ki koshish.
- Linear 128 -> 5: har class ka raw score (logit).

Training mein output par Softmax layer nahi lagate: CrossEntropyLoss internally log-softmax handle karta hai. Prediction ke waqt torch.softmax scores ko 5 probabilities mein convert karta hai.

## 4. Model seekhta kaise hai?
Forward pass prediction scores banata hai. CrossEntropyLoss true label se comparison karta hai. loss.backward gradients nikalta hai. Adam optimizer weights update karta hai. Ek epoch mein har training image ek baar dekhi jati hai. Ye process 30 epochs default repeat hota hai. Best validation accuracy wala model save hota hai; test accuracy model choose karne ke liye use nahi hoti.

## 5. Accuracy aur confusion matrix
Accuracy = correct predictions / total images. Per-class accuracy se pata chalta hai har insect par kitna sahi hai. Confusion matrix ki rows actual class aur columns predicted class hain. Diagonal correct; off-diagonal mistakes. Curves mein train improve lekin validation loss barhe to overfitting ka signal ho sakta hai. Test curves assignment ke mutabiq log hoti hain; un par bar bar tuning nahi karni.

## 6. Database
users table: id, unique username, salt, password_hash, created_at.
image_records: id, user_id, image_path, predicted_label, confidence, optional true_label, created_at.
users ka aik row multiple image_records se link hota hai: one-to-many relationship. user_id foreign key hai. Har query logged-in user ka id use karti hai, is liye doosre user ki saved history show nahi hoti.

Password direct store nahi hota. Random salt + PBKDF2 hashing use hoti hai. Login par entered password ka same salt se hash banta hai aur compare hota hai. SQL placeholders input ko SQL code banne se rokte hain.

## 7. GUI
Login/Register -> Predict -> image preview -> top-3 results -> Save -> My History -> preview/delete/export -> Logout. Save par original file ki RGB PNG copy app_data/records mein rakhi jati hai. Original source image delete nahi hoti. Model startup par train nahi hota; sirf trained checkpoint prediction ke liye load hota hai.

## Viva: 12 short answers
1. CNN kyun? Image ke local/spatial patterns seekhta hai.
2. 5 outputs kyun? Assigned classes 5 hain.
3. RGB kya hai? Red, green, blue channels.
4. Validation kyun? Checkpoint select karna, test ko selection se bachana.
5. Pretrained use hua? Nahi; weights scratch se train hote hain.
6. Dropout kyun? Overfitting reduce karne ke liye regularization.
7. Model file? models/insect_cnn.pt; state_dict + class metadata.
8. Confidence? Softmax estimate; guarantee nahi.
9. History private kaise? user_id scoped SELECT/DELETE.
10. SQLite kyun? Small local app ke liye single-file relational database.
11. Non-insect image? Phir bhi 5 mein se aik class choose karega; unknown detector nahi.
12. Improvement? More representative data, validation-based hyperparameter tuning, confidence calibration; assignment ki allowed layers ke andar.

Apne actual report.pdf aur outputs/metrics.json mein measured results dekhein. Koi accuracy ya explanation yaad se invent na karein. Misclassified images ki possible visual wajah ko hypothesis kahen, proven explanation nahi.
