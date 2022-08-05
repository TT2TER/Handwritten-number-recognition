import torch
from net import LeNet
import torchvision.transforms as trans
import cv2 as cv

the_net = LeNet()
cam=1
trans1=trans.ToTensor()
the_net=torch.load('./model/model_11.pth')
cap = cv.VideoCapture(0)

def maxi_area(frame):
        gray=cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        img = cv.GaussianBlur(gray,(9,13),0)
        canny = cv.Canny(img, 50, 150)
        contours, hierarchy = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        dot=[]
        
        for c in contours:
            min_list=[] 
            x, y, w, h = cv.boundingRect(c) 
            min_list.append(x)
            min_list.append(y)
            min_list.append(w)
            min_list.append(h)
            min_list.append(w*h)
            dot.append(min_list)
        
        max_area=dot[0][4] 
        for inlist in dot:
            area=inlist[4]
            if area >= max_area:
                x=inlist[0]
                y=inlist[1]
                w=inlist[2]
                h=inlist[3]
                max_area=area
        
        cv.rectangle(frame, (x, y), (x + w , y + h ), (0, 255, 0), 1)
        cv.imshow('zoon',frame[y:y+h,x:x+w])

        if h>w:
            number_zoon=frame[((2*y+h)//2)-(w//2)+5:((2*y+h)//2)+(w//2)-5,x:x+w]
        else:
            number_zoon=frame[y+5:y+h-5,((2*x+w)//2)-(h//2):((2*x+w)//2)+(h//2)]
        return number_zoon


def see(number_zoon):
            compress=cv.resize(number_zoon,(28,28),interpolation=cv.INTER_CUBIC)
            img = cv.GaussianBlur(compress,(3,3),0)
            ret,fan=cv.threshold(img,190,255,cv.THRESH_BINARY_INV)
            
            cv.imshow('input',fan)

            x=trans1(fan)
            x=torch.unsqueeze(x, dim=0)

            result=the_net(x)
            values, res = torch.max(result, 1)
            res=res.numpy()
            values=values.detach().numpy()
            return values,res

def localEqualHist(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2, tileGridSize=(8,8))
    dst = clahe.apply(gray)
    return dst


while(cap.isOpened()):
    success, frame = cap.read()
    if(cam):
        print('启动成功!摄像头宽高为',cap.get(3),'*',cap.get(4))
        cam=0
    if success:
        number_zoon=maxi_area(frame)
        number_zoon=localEqualHist(number_zoon)
        number_zoon = cv.GaussianBlur(number_zoon,(3,3),0)
        values,res=see(number_zoon)
        
        cv.putText(frame, 'press q to quit', (20,100),cv.FONT_ITALIC, 0.7, (255, 255, 128), 2, cv.LINE_AA)
        
        if values>=0.99:
            text=('ans=%d  acc=%f'% (res,values))
            cv.putText(frame, text, (20,60),cv.FONT_ITALIC, 1.0, (128, 0, 128), 2, cv.LINE_AA)
            cv.imshow('img',frame)
        else:
            text=('No number detected')
            cv.putText(frame, text, (20,60),cv.FONT_ITALIC, 1.0, (0, 0, 255), 2, cv.LINE_AA)
            cv.imshow('img',frame)
    else:
        break
        
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()