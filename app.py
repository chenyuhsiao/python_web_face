#使用教學
#請先利用ipconfig查詢ip，並在最後新增app.run(host=IP,port=80)
#網站圖片儲存的位置請選擇檔案位置




#網頁傳值用套件
from werkzeug.utils import secure_filename
from flask import Flask, render_template, jsonify, request, make_response, send_from_directory, abort,redirect, url_for
import os
from PIL import Image 
#臉部辨識套件
import face_recognition
import cv2
import numpy as np

#臉部辨識程式
face_encoding_list=[]
known_face_name=[]
#biden_img = face_recognition.load_image_file("biden.jpg")
#obama_img = face_recognition.load_image_file("obama.jpg")
#brian_img = face_recognition.load_image_file("brian.jpg")
#trump_img = face_recognition.load_image_file("trump.jpg")
#biden_encoding = face_recognition.face_encodings(biden_img)[0]
#obama_encoding = face_recognition.face_encodings(obama_img)[0]
#brian_encoding = face_recognition.face_encodings(brian_img)[0]
#trump_encoding = face_recognition.face_encodings(trump_img)[0]
#face_encoding_list=[biden_encoding,obama_encoding,brian_encoding,trump_encoding]
#known_face_name=["biden","obama","brian","trump"]

#np.save("data_encoding",face_encoding_list,allow_pickle=True, fix_imports=True)
#np.load("data_encoding.npy")
face_encoding_list=np.load("data_encoding.npy")

#np.save("data_name",known_face_name,allow_pickle=True, fix_imports=True)
#np.load("data_name.npy")
known_face_name=np.load("data_name.npy")

#網站圖片儲存位置
UPLOAD_FOLDER = '/Users/chenyuxiao/Desktop/app1_server'
RECOGNIZE_FLODER='/Users/chenyuxiao/Desktop/app1_server'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'gif'])

#網站程式
app=Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RECOGNIZE_FLODER'] = RECOGNIZE_FLODER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.config['JSON_AS_ASCII'] = False

basedir = os.path.abspath(os.path.dirname(__file__))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route("/", methods=['GET', 'POST'])
def index():

    return render_template('server_face_recognition.html')
#upload_pic
##臉部訓練##

@app.route('/upload_pic', methods=['GET','POST'])
def upload_pic():
    #建立空陣列
    train_encoding_data=[]
    check_count=1
    file_save_count=2
    data_name=np.load("data_name.npy")
    data_encoding=np.load("data_encoding.npy")


    data_name_count=len(data_name)
    if request.values['upload']=='清除訓練':
        password=request.values["password"]
        if password=="0112":
            data_name=[]
            data_encoding=[]
            
            np.save("data_name",data_name,allow_pickle=True,fix_imports=True)
            np.save("data_encoding",data_encoding,allow_pickle=True,fix_imports=True)
            output1="清除訓練"
        else:
            output1="密碼錯誤"

    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename ="train.jpeg" #secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_save_count=1
        
        train_img=face_recognition.load_image_file("train.jpeg")
        train_face_locations=face_recognition.face_locations(train_img)
        train_locations_counts=len(train_face_locations)

        if request.values['upload']=='創建訓練':
            train_name=request.values["user"]
            print("訓練",train_name)
            if train_name=="":
                check_count=2
                output1="請輸入訓練名稱"
            else:
                check_count=1
                if train_locations_counts==1 and file_save_count==1:
                    if check_count==1:
                        for i in range(data_name_count):
                            if data_name[i]==str(train_name):
                                output1="訓練重複"
                                check_count=2
                    if check_count==1:
                        #訓練名字的部分
                        output1="成功訓練"+str(train_name)  
                        data_name=np.append(data_name,train_name)
                    
                        #訓練encoding的部分
                        train_face_encoding=face_recognition.face_encodings(train_img)[0]
                        data_encoding=np.load("data_encoding.npy")
                        for i in range(data_name_count):
                            train_encoding_data.append(data_encoding[i])

                        train_encoding_data.append(train_face_encoding)
                        print(train_encoding_data)

                        #encoding[data_name_count]=train_face_encoding
                        #print(encoding[data_name_count])                  
                        #data_encoding.append(train_face_encoding)                   
                        #data_encoding=np.append(data_encoding,train_face_encoding)
                    
                    #清除訓練
                    #data_name=[]
                    #data_encoding=[]
                    np.save("data_name",data_name,allow_pickle=True,fix_imports=True)
                    np.save("data_encoding",train_encoding_data,allow_pickle=True,fix_imports=True)
                else:
                    output1="訓練失敗"
    data_name=np.load("data_name.npy")
    data_encoding=np.load("data_encoding.npy")


    print("npy目前儲存的人名",data_name)
    print("輸出結果:",output1)

    return render_template('server_face_recognition.html',output1=output1)

##臉部辨識##
@app.route('/recognize_pic', methods=['GET', 'POST'])
def recognize_pic():
    if request.method=="POST":
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename ="test.jpeg" #secure_filename(file.filename)
            file.save(os.path.join(app.config['RECOGNIZE_FLODER'], filename))

#臉部辨識程式
    unknown_img = face_recognition.load_image_file("test.jpeg")
    unknown_face_locations = face_recognition.face_locations(unknown_img) 
    location_count=len(unknown_face_locations)
    location_true=False
    if location_count>=1:
        location_true=True
    print("圖片中總共有",location_count,"張臉")

    save_count=0
    output_name=[]

#儲存圖片中每一個臉
    for face_location in unknown_face_locations:
        top, right, bottom, left = face_location  
        #print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))    
        face_image = unknown_img[top:bottom, left:right]  
        pil_image = Image.fromarray(face_image)  
        #pil_image.show()

        save_count=save_count+1
        save_name="test"+str(save_count)+".jpeg"
        pil_image.save(save_name)

#針對每個臉做compare
        face_encoding_list=np.load("data_encoding.npy")
        known_face_name=np.load("data_name.npy")

        image=face_recognition.load_image_file(save_name)
        unknown_face_encoding=face_recognition.face_encodings(image)[0]
        results=face_recognition.compare_faces(face_encoding_list,unknown_face_encoding)
        #print(results[0])
        results_count=len(results)
#標示人名:
        for i in range(results_count):
            if str(results[i])=="True":
                output_name.append(known_face_name[i])
        if output_name==[]:
            output_name="  沒有認識的資料"

        #else:
            #output_name.append("unknown")

    #unknown_face_encoding = face_recognition.face_encodings(unknown_img)[0]
    #results = face_recognition.face_distance(face_encoding_list,unknown_face_encoding)

    #march=0
    #output1=[]
   # p=0
   # for (i, r) in enumerate(results):
        #if r==0:
            #march=100
            #output1=[known_face_name[i]]
            #output1.append(known_face_name[i])
        #else:
            #p=100-round(r*100)
            #if p>march:
                #march=p
                #if march>=0:
                    #output1=[known_face_name[i]]
                    #output1.append(known_face_name[i])
    json_result={
        "face_found":location_true,
        "face_of_":output_name
    }

    return render_template('server_face_recognition.html',output=output_name,json_result=json_result)

    #return render_template('index.html',name1=request.values["user"])
      #      return redirect(url_for('recognize',filename=filename))

if __name__=="__main__":

    app.run(debug=True)
