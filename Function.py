import cv2
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab as pl


# Define class to store enclosing rectangle's information:
class Element:
    Location=(int, int, int, int)


# Define function to generate temporary image:
def GenerateTemporaryImage(InputPath, OutputPath, Ruler=0):
    Img=cv2.imread(InputPath)
    if Ruler==1:
        H, W=Img.shape[:2]
        Img=Img[200:H, 0:W]
    cv2.imwrite(OutputPath, Img)


# Define function to enhance edges:
def StrokeEdges(Img, OutputPath, BlurSize=7, EnhanceSize=5):
    # Erase noise:
    if BlurSize>=3:
        ImgBlur=cv2.medianBlur(Img.copy(), BlurSize)
        ImgBlurGray=cv2.cvtColor(ImgBlur.copy(), cv2.COLOR_BGR2GRAY)
    else:
        ImgBlurGray=cv2.cvtColor(Img.copy(), cv2.COLOR_BGR2GRAY)
    
    # Enhance edges:
    cv2.Laplacian(ImgBlurGray, cv2.CV_8U, ImgBlurGray, EnhanceSize)
    Coefficient=(1.0/255)*(255-ImgBlurGray)
    Channels=cv2.split(Img)
    for Channel in Channels:
        Channel[:]=Channel*Coefficient
        cv2.merge(Channels, Img)
    cv2.imwrite(OutputPath, Img)


# Define function to extract currency contour:
def ExtractContour(InputPath, OutputPath, OriginalPath):
    # Path needed to save temporary image:
    ImageThresh="D:\\DATABASE\\Thresh.jpg"
    ImageConnection="D:\\DATABASE\\Connection.jpg"
    ImageContour="D:\\DATABASE\\Contour.jpg"
    
    # Read image in gray scale:
    ImgGray=cv2.imread(InputPath, cv2.IMREAD_GRAYSCALE)
    
    # Threshhold of gray scale image:
    ret, thresh=cv2.threshold(ImgGray.copy(), 200, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite(ImageThresh, thresh)
    
    # Eliminate upper-side effect:
    thresh[0:10, :]=0
    
    # Connect adjacent area using kernel below:
    Kernel=np.array([[1, 1, 1, 1, 1], 
                        [1, 1, 1, 1, 1], 
                        [1, 1, 1, 1, 1], 
                        [1, 1, 1, 1, 1], 
                        [1, 1, 1, 1, 1]])
    cv2.filter2D(thresh, -1, Kernel, thresh)
    cv2.imwrite(ImageConnection, thresh)
    
    # Find contours:
    image, contours, hier=cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Fiind the largest enclosing rectangle:
    ImgContour=cv2.imread(OriginalPath)
    X=0
    Y=0
    W=0
    H=0
    for c in contours:
        x,y,w,h=cv2.boundingRect(c)
        if w>W and h>H:
            X=x
            Y=y
            W=w
            H=h
    cv2.rectangle(ImgContour, (X, Y-8), (X+W, Y+H), (0, 255, 0), 2)
    cv2.imwrite(ImageContour, ImgContour)
    
    # Extract curency contour:
    ImgNote=cv2.imread(OriginalPath)[Y-8:Y+H, X:X+W]
    cv2.imwrite(OutputPath, ImgNote)


# Define function to normalize note:
def Normalize(InputPath, OutputPath, BlurSize=99, ResizeH=50, ResizeW=100):
    ImgNote=cv2.imread(InputPath)
    ImgNoteBlur=cv2.medianBlur(ImgNote.copy(), BlurSize)
    ImgNoteBlurGray=cv2.cvtColor(ImgNoteBlur, cv2.COLOR_BGR2GRAY)
    ImgNoteBlurGrayNormalized=cv2.resize(ImgNoteBlurGray, (ResizeW, ResizeH))
    cv2.imwrite(OutputPath, ImgNoteBlurGrayNormalized)


# Define function to generate note image from original image:
def IdentifyNote(InputPath, OutputPath, Ruler):
    # Path needed to store temporary image:
    ImageOriginal="D:\\DATABASE\\Original.jpg"
    ImageStroked="D:\\DATABASE\\Stroked.jpg"
    
    # Use functions defined above to identify note:
    GenerateTemporaryImage(InputPath, ImageOriginal, Ruler)
    Img=cv2.imread(ImageOriginal)
    StrokeEdges(Img, ImageStroked, 7, 11)
    ExtractContour(ImageStroked, OutputPath, ImageOriginal)


# Define function to compare the difference between two image:
def Compare(InputPath, AbstractPath, AbstractName, Method=0):
    InputNumpy=cv2.imread(InputPath, cv2.IMREAD_GRAYSCALE)
    InputNumpy=np.int16(InputNumpy.flatten())
    AbstractNumpy=cv2.imread(AbstractPath, cv2.IMREAD_GRAYSCALE)
    AbstractNumpy=np.int16(AbstractNumpy.flatten())
    # Compare with standard deviation:
    if Method==0:
        Result=abs(AbstractNumpy-InputNumpy-(np.average(AbstractNumpy)-np.average(InputNumpy)))
        ResultStd=np.std(Result)
        return ResultStd
    # Compare with average:
    else:
        Result=abs(AbstractNumpy-InputNumpy-(np.average(AbstractNumpy)-np.average(InputNumpy)))
        ResultAverage=np.average(Result)
        return ResultAverage


# Define function to recognize currency's nation:
def RecognizeNation(ImageInput):
    # Database path:
    ImageAbstractEUR=("D:\\DATABASE\\CurrencyCategory\\Abstract\\EUR.jpg")
    ImageAbstractGBP_NEW=("D:\\DATABASE\\CurrencyCategory\\Abstract\\GBP-NEW.jpg")
    ImageAbstractGBP_OLD=("D:\\DATABASE\\CurrencyCategory\\Abstract\\GBP-OLD.jpg")
    ImageAbstractHKD_OTHER=("D:\\DATABASE\\CurrencyCategory\\Abstract\\HKD-OTHER.jpg")
    ImageAbstractHKD_TEN=("D:\\DATABASE\\CurrencyCategory\\Abstract\\HKD-TEN.jpg")
    ImageAbstractJPY=("D:\\DATABASE\\CurrencyCategory\\Abstract\\JPY.jpg")
    ImageAbstractKRW=("D:\\DATABASE\\CurrencyCategory\\Abstract\\KRW.jpg")
    ImageAbstractRMB=("D:\\DATABASE\\CurrencyCategory\\Abstract\\RMB.jpg")
    ImageAbstractTWD=("D:\\DATABASE\\CurrencyCategory\\Abstract\\TWD.jpg")
    ImageAbstractUSD=("D:\\DATABASE\\CurrencyCategory\\Abstract\\USD.jpg")
    
    # Use functions defined above to compare:
    EUR=Compare(ImageInput, ImageAbstractEUR, "EUR")
    GBP_NEW=Compare(ImageInput, ImageAbstractGBP_NEW, "GBP-NEW")
    GBP_OLD=Compare(ImageInput, ImageAbstractGBP_OLD, "GBP-OLD")
    HKD_OTHER=Compare(ImageInput, ImageAbstractHKD_OTHER, "HKD-OTHER")
    HKD_TEN=Compare(ImageInput, ImageAbstractHKD_TEN, "HKD-TEN")
    JPY=Compare(ImageInput, ImageAbstractJPY, "JPY")
    KRW=Compare(ImageInput, ImageAbstractKRW, "KRW")
    RMB=Compare(ImageInput, ImageAbstractRMB, "RMB")
    TWD=Compare(ImageInput, ImageAbstractTWD, "TWD")
    USD=Compare(ImageInput, ImageAbstractUSD, "USD")
    
    # Find the most similar one:
    List=[EUR, GBP_NEW, GBP_OLD, HKD_OTHER, HKD_TEN, JPY, KRW, RMB, TWD, USD]
    ListMin=List.copy()
    list.sort(ListMin)
    if (ListMin[0]/ListMin[1])<1:
        Rank=List.index(ListMin[0])
        if Rank==0:
            print("TYPE:          EUR")
        elif Rank==1:
            print("TYPE:          GBP")
        elif Rank==2:
            print("TYPE:          GBP")
        elif Rank==3:
            print("TYPE:          HKD")
        elif Rank==4:
            print("TYPE:          HKD")
        elif Rank==5:
            print("TYPE:          JPY")
        elif Rank==6:
            print("TYPE:          KRW")
        elif Rank==7:
            print("TYPE:          RMB")
        elif Rank==8:
            print("TYPE:          TWD")
        elif Rank==9:
            print("TYPE:          USD")
    else:
        Rank=-1
    return Rank


# Define function to extract specific kind of contours:
def ExtractDetail(InputPath, List, BlurValue, StrokeValue, ThreshValue, UpperH, LowerH, UpperW, LowerW, UpperR, LowerR):
    # Path needed to store temporary image:
    ImageNoteStroked="D:\\DATABASE\\NoteStroked.jpg"
    ImageNoteThresh="D:\\DATABASE\\NoteThresh.jpg"
    
    # Extract contours:
    ImgNote=cv2.imread(InputPath)
    H, W=ImgNote.shape[:2]
    StrokeEdges(ImgNote, ImageNoteStroked, BlurValue, StrokeValue)
    ImgNoteStroked=cv2.imread(ImageNoteStroked, cv2.IMREAD_GRAYSCALE)
    ret, thresh=cv2.threshold(ImgNoteStroked.copy(), ThreshValue, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite(ImageNoteThresh, thresh)
    image, contours, hier=cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    i=0
    for c in contours:
        x,y,w,h=cv2.boundingRect(c)
        if h in range(int(H*LowerH), int(H*UpperH)) and w in range(int(W*LowerW), int(W*UpperW))\
        and w/h>LowerR and w/h<UpperR and x+w!=W and y+h!=H and x!=0 and y!=0:
            Name="Element"+str(i)
            Name=Element()
            List.append(Name)
            List[i].Location=(x, y, w, h)
            i=i+1
    return List


# Define function to connect and search series of contours' enclosing rectangle:
def SearchSeries(InputPath, List, ListSeries, ExtIntH, ExtIntW, UpperSeriesR, LowerSeriesR):
    # Path needed to store temporary image:
    ImageNoteTemp="D:\\DATABASE\\NoteTemp.jpg"
    ImageSeries="D:\\DATABASE\\Series.jpg"
    
    # Connect enclosing rectangles:
    Img=cv2.imread(InputPath)
    H, W=Img.shape[:2]
    thresh=np.zeros((H, W), dtype=np.uint8)
    for i in range(0, len(List)):
        x=List[i].Location[0]
        y=List[i].Location[1]
        w=List[i].Location[2]
        h=List[i].Location[3]
        cv2.rectangle(thresh, (x-ExtIntW, y-ExtIntH), (x+w+ExtIntW, y+h+ExtIntH), 255, 2)
    cv2.imwrite(ImageNoteTemp, thresh)
    
    # Find contours from connection image:
    image, contours, hier=cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    WMax=0
    for c in contours:
        X, Y, W, H=cv2.boundingRect(c)
        if W/H>LowerSeriesR and W/H<UpperSeriesR:
            if W>WMax:
                del ListSeries[:]
                WMax=W
                for i in range(0, len(List)):
                    x=List[i].Location[0]
                    y=List[i].Location[1]
                    w=List[i].Location[2]
                    h=List[i].Location[3]
                    if x in range(X, X+W) and y in range(Y, Y+H):
                        ListSeries.append(List[i])
                        
    # Return list storing enclosing rectangles' information:
    return ListSeries
    
    # Draw enclosing rectangle when needed:
    for i in range(0, len(ListSeries)):
        x=ListSeries[i].Location[0]
        y=ListSeries[i].Location[1]
        w=ListSeries[i].Location[2]
        h=ListSeries[i].Location[3]
        cv2.rectangle(Img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imwrite(ImageSeries, Img)


# Define function to re-sort enclosing rectangle information list:
def RearrangeListInX(List, ListOut, Y=0):
    XList=[]
    for i in range(0, len(List)):
        XList.append(List[i].Location[Y])
    XSortList=XList.copy()
    list.sort(XSortList)
    for i in range(0, len(List)):
        ListOut.append(List[XList.index(XSortList[i])])
    return ListOut
    

# Define funtion to cut and normalize element from original image:
def NormalizeElement(InputPath, List):
    # Path needed to store temporary image:
    ImageElement="D:\\DATABASE\\Element"
    
    # Cut and normalize element from original image:
    Img=cv2.imread(InputPath, cv2.IMREAD_GRAYSCALE)
    H, W=Img.shape[:2]
    for i in range(0, len(List)):
        x=List[i].Location[0]
        y=List[i].Location[1]
        w=List[i].Location[2]
        h=List[i].Location[3]
        MidX=int((2*x+w)/2)
        MidY=int((2*y+h)/2)
        R=int(max(w, h)/2)
        Name=np.zeros((H, W), dtype=np.uint8)
        Name[y:y+h, x:x+w]=Img[y:y+h, x:x+w]
        Left=MidX-R
        Right=MidX+R
        if Left<0:
            Left=0
        if Right>=W:
            Right=W
        Norm=Name[MidY-R:MidY+R, Left:Right]
        Norm=cv2.resize(Norm, (20, 20))
        ImageElements=ImageElement+str(i)+".jpg"
        cv2.imwrite(ImageElements, Norm)


# Define funtion to calculate curve's deviation value from its peak:
def AnalyzeCurve(InputNumpy, CurveName):
    X=InputNumpy
    X=np.int32(X)
    
    # Find curve's peak:
    Max=np.max(X)
    Base=np.argwhere(X==Max)
    Base=int(Base)
    
    # Calculate area:
    Area=0
    for i in range(1, len(X)-1):
        Area=Area+((i-Base))*X[i]
    return(Area)
    
    # Draw curve when needed:
    df=pd.DataFrame(X, columns=['X'])
    df[CurveName]=pd.Series(list(range(len(df))))
    plt.figure()
    df.plot(CurveName)
    plt.show()

    
# Define funtion to extract dominant color detail from a given image:
def ExtractColorDetail(InputPath, Function=1):
    Img=cv2.imread(InputPath)
    H, W=Img.shape[:2]
    
    # numpy to store different color channel's information
    ch0=np.zeros((H, W))
    ch1=np.zeros((H, W))
    ch2=np.zeros((H, W))
    channels=cv2.split(Img)
    i=0
    for channel in channels:
        if i==0:
            ch0[:]=channel[:]
        if i==1:
            ch1[:]=channel[:]
        if i==2:
            ch2[:]=channel[:]
        i=i+1
    cv2.merge(channels, Img)
    
    # Optional funtion to return image's other color information:
    if Function==0:
        AverageB=round(np.average(ch0))
        AverageG=round(np.average(ch1))
        AverageR=round(np.average(ch2))
        StdB=round(np.std(ch0))
        StdG=round(np.std(ch1))
        StdR=round(np.std(ch2))
        return(AverageB, AverageG, AverageR, StdB, StdG, StdR)
    
    # Main funtion:
    chB=ch0/(ch0+ch1+ch2)
    chG=ch1/(ch0+ch1+ch2)
    chR=ch2/(ch0+ch1+ch2)
    B=chB.flatten()
    G=chG.flatten()
    R=chR.flatten()
    B=np.int8(B*100)
    G=np.int8(G*100)
    R=np.int8(R*100)
    X=np.bincount(B)
    Y=np.bincount(G)
    Z=np.bincount(R)
    
    # Calculate deviation value:
    BWeigth=AnalyzeCurve(X, "B")
    GWeigth=AnalyzeCurve(Y, "G")
    RWeigth=AnalyzeCurve(Z, "R")
    
    # Resize deviation value and return:
    return([int(BWeigth*20/(H*W)), int(GWeigth*20/(H*W)), int(RWeigth*20/(H*W))])


# Define funtion to convert 3-dimension information to 2-dimension:
def ConvertCoordinate(B, G, R):
    SX=1
    SY=1
    if B>G:
        SX=-1
    if R<B+G:
        SY=-1
    X=round(SX*math.sqrt((B-G)*(B-G)/2))
    Y=round(SY*math.sqrt((R*R/2)+(-(2/3)*B-(2/3)*G+(1/3)*R)**2))
    return(X, Y)


# Define funtion to recognize number 1, 2, 5:
def RecognizeFacevalueNumber(FacevalueElementPath, Length):
    # Compare image with database image:
    ImageAbstract="D:\\DATABASE\\CurrencyFacevalue\\Number\\"
    if Length==0:
        return -1
    CompareList=[]
    for i in range (1, 4):
        AbstractPath=ImageAbstract+str(i)+".jpg"
        CompareList.append(Compare(FacevalueElementPath, AbstractPath, str(i)))
    
    # Find the most similar one and return:
    ListMin=CompareList.copy()
    list.sort(ListMin)
    Rank=CompareList.index(ListMin[0])
    if Rank==0:
        return 1*10**(Length-1)
    elif Rank==1:
        return 2*10**(Length-1)
    else:
        return 5*10**(Length-1)


# Define funtion to recognize face value:
def RecognizeFacevalue(InputPath, CurrencyType):
    # Path needed to store temporary image:
    ImageFacevalueROI="D:\\DATABASE\\FacevalueROI.jpg"
    ImageNoteThresh="D:\\DATABASE\\NoteThresh.jpg"
    ImageFacevalueROINormalized="D:\\DATABASE\\FacevalueROINormalized.jpg"
    ImageFacevalueElement="D:\\DATABASE\\Element0.jpg"
    
    # Load inmage:
    Img=cv2.imread(InputPath)
    H, W=Img.shape[:2]
    
    # List needed to store temporary information:
    Set=[]
    SetSeries=[]
    SetInX=[]
    
    # Face value recognition for EUR:
    if CurrencyType==0:
        Img=Img[int(H*(2/3)):H, 0:int(W*(1/4))]
        cv2.imwrite(ImageFacevalueROI, Img)
        
        ExtractDetail(ImageFacevalueROI, Set, 1, 1, 200, 2/3, 1/3, 1, 0, 1, 1/4)
        SearchSeries(ImageFacevalueROI, Set, SetSeries, 0, 10, 4, 1/2)
        RearrangeListInX(SetSeries, SetInX)
        NormalizeElement(ImageNoteThresh, SetInX)
        
        Length=len(SetInX)
        Facevalue=RecognizeFacevalueNumber(ImageFacevalueElement, Length)
        return Facevalue
    
    # Face value recognition for GBP-NEW:
    if CurrencyType==1:
        Img=Img[0:int(H*(1/4)), int(W*(4/5)):W]
        cv2.imwrite(ImageFacevalueROI, Img)
        
        ExtractDetail(ImageFacevalueROI, Set, 1, 1, 175, 2/3, 1/3, 1, 0, 1, 1/4)
        SearchSeries(ImageFacevalueROI, Set, SetSeries, 0, 10, 4, 1)
        RearrangeListInX(SetSeries, SetInX)
        del SetInX[0]
        NormalizeElement(ImageNoteThresh, SetInX)
        
        Length=len(SetInX)
        Facevalue=RecognizeFacevalueNumber(ImageFacevalueElement, Length)
        return Facevalue
    
    # Face value recognition for GBP-OLD:
    if CurrencyType==2:
        Img=Img[0:int(H*(2/7)), int(W*(6/7)):W]
        cv2.imwrite(ImageFacevalueROI, Img)
        
        ExtractDetail(ImageFacevalueROI, Set, 1, 1, 175, 2/3, 1/3, 1, 0, 1, 1/6)
        SearchSeries(ImageFacevalueROI, Set, SetSeries, 0, 10, 2, 2/5)
        RearrangeListInX(SetSeries, SetInX)
        NormalizeElement(ImageNoteThresh, SetInX)
        
        Length=len(SetInX)
        Facevalue=RecognizeFacevalueNumber(ImageFacevalueElement, Length)
        return Facevalue
    
    # Face value recognition for HKD-OTHER:
    if CurrencyType==3:
        Area=round(H*W/10000)
        if Area in range(120, 154):
            return 20
        elif Area in range(154, 162):
            return 50
        elif Area in range(162, 178):
            return 100
        elif Area in range(178, 189):
            return 500
        elif Area in range(189, 220):
            return 1000
        else:
            return -1
    
    # Face value recognition for HKD-10:
    if CurrencyType==4:
        return 10
    
    # Face value recognition for JPY:
    if CurrencyType==5:
        Area=round(H*W/10000)
        if Area in range(150, 171):
            return 1000
        elif Area in range(171, 177):
            return 5000
        elif Area in range(177, 200):
            return 10000
        else:
            return -1
    
    # Face value recognition for KRW:
    if CurrencyType==6:
        Img=Img[0:int(H*(2/3)), int(W*(7/8)):int(W*(23/24))]
        Img=cv2.transpose(Img, Img)
        Img=cv2.flip(Img, 0)
        cv2.imwrite(ImageFacevalueROI, Img)
        
        ExtractDetail(ImageFacevalueROI, Set, 1, 1, 150, 1, 3/5, 1, 0, 1, 1/6)
        SearchSeries(ImageFacevalueROI, Set, SetSeries, 0, 10, 5, 2)
        RearrangeListInX(SetSeries, SetInX)
        NormalizeElement(ImageNoteThresh, SetInX)
        
        Length=len(SetInX)
        Facevalue=RecognizeFacevalueNumber(ImageFacevalueElement, Length)
        return Facevalue
    
    # Face value recognition for RMB:
    if CurrencyType==7:
        B, G, R=ExtractColorDetail(InputPath)
        X, Y=ConvertCoordinate(B, G, R)
        
        if X in range(45, 80) and Y in range(0, 50):
            return 1
        elif X in range(-30, 0) and Y in range(0, 40):
            return 5
        elif X in range(-20, 10) and Y in range(-50, 0):
            return 10
        elif X in range(10, 45) and Y in range(10, 100):
            return 20
        elif X in range(10, 50) and Y in range(-70, -10):
            return 50
        elif X in range(-40, 10) and Y in range(75, 200):
            return 100
        else:
            return -1
    
    # Face value recognition for TWD:
    if CurrencyType==8:
        if W in range(1650, 1777):
            return 100
        elif W in range(1777, 1842):
            return 200
        elif W in range(1842, 1894):
            return 500
        elif W in range(1894, 1958):
            return 1000
        elif W in range(1958, 2100):
            return 2000
        else:
            return -1
    
    # Face value recognition for USD:
    if CurrencyType==9:
        Img=Img[int(H*(1/6)):int(H*(5/6)), int(W*(1/3)):int(W*(2/3))]
        cv2.imwrite(ImageFacevalueROI, Img)
        
        Normalize(ImageFacevalueROI, ImageFacevalueROINormalized, 15, 30, 30)
        ImageAbstract="D:\\DATABASE\\CurrencyFacevalue\\USD\\"
        ImageMask="D:\\DATABASE\\CurrencyFacevalue\\USD\\Mask.jpg"
        CompareList=[]
        
        for i in range(1, 17):
            AbstractPath=ImageAbstract+str(i)+".jpg"
            CompareList.append(Compare(ImageFacevalueROINormalized, AbstractPath, str(i), 1))
        
        ListMin=CompareList.copy()
        list.sort(ListMin)
        Rank=CompareList.index(ListMin[0])
        if Rank==0:
            return 1
        elif Rank==1:
            return 2
        elif Rank in range(3, 5):
            return 5
        elif Rank in range(6, 8):
            return 10
        elif Rank in range(9, 11):
            return 20
        elif Rank in range(11, 14):
            return 50
        elif Rank==2 or Rank==5 or Rank==8:
            Img=cv2.imread(ImageFacevalueROINormalized)
            ImgNumpy=np.int16(Img.flatten())
            Median=np.average(ImgNumpy)
            ret, thresh=cv2.threshold(Img.copy(), Median, 255, cv2.THRESH_BINARY)
            thresh=thresh[19:28, 7:24]
            cv2.imwrite(ImageFacevalueROINormalized, thresh)
            
            CompareList=[]
            for i in range(21, 24):
                AbstractPath=ImageAbstract+str(i)+".jpg"
                InputNumpy=thresh
                InputNumpy=np.int16(InputNumpy.flatten())
                AbstractNumpy=cv2.imread(AbstractPath)
                AbstractNumpy=np.int16(AbstractNumpy.flatten())
                MaskNumpy=cv2.imread(ImageMask)
                MaskNumpy=np.int16(MaskNumpy.flatten())
                Result=np.sum((InputNumpy==AbstractNumpy)*(InputNumpy!=0)*MaskNumpy)
                CompareList.append(Result)
                
            ListMin=CompareList.copy()
            list.sort(ListMin)
            Rank=CompareList.index(ListMin[2])
            if Rank==0:
                return 5
            elif Rank==1:
                return 10
            else:
                return 20
        else:
            return 100


# Define filter with dominant color:
def ColorFilter(InputPath, OutputPath, SubColorPath, Value=128):
    # Analyze curve's information:
    DeviateB, DeviateG, DeviateR=ExtractColorDetail(InputPath)
    
    # Fine dominant color:
    BGRList=[DeviateB, DeviateG, DeviateR]
    BGRListMin=BGRList.copy()
    list.sort(BGRListMin)
    RankMax=BGRList.index(BGRListMin[2])
    
    # Filter:
    Img=cv2.imread(InputPath)
    channels=cv2.split(Img)
    i=0
    for channel in channels:
        if RankMax==i:
            channel[:]=channel[:]+Value
            ImgSub=-channel[:]
        i=i+1
    cv2.merge(channels, Img)
    cv2.imwrite(OutputPath, Img)
    cv2.imwrite(SubColorPath, ImgSub)


# Define function to recognize number:
def RecognizeNumber(InputPath):
    # Numer database path:
    ImageAbstract="D:\\DATABASE\\CurrencyCharacter\\"
    
    # Comparing input image with database image:
    CompareList=[]
    for i in range(0, 10):
        AbstractPath=ImageAbstract+str(i)+".jpg"
        Std=Compare(InputPath, AbstractPath, str(i), 1)
        CompareList.append(Std)
    for i in range(100, 110):
        AbstractPath=ImageAbstract+str(i)+".jpg"
        Std=Compare(InputPath, AbstractPath, str(i), 1)
        CompareList.append(Std)
    
    # Find the most similar one and return:
    ListMin=CompareList.copy()
    list.sort(ListMin)
    Rank=CompareList.index(ListMin[0])
    for i in range(0, 10):
        if Rank==i:
            return chr(i+48), CompareList[i]
    if Rank==10:
        return "2", CompareList[Rank]
    elif Rank==11 or Rank==19:
        return "3", CompareList[Rank]
    elif Rank==12 or Rank==15:
        return "6", CompareList[Rank]
    elif Rank==13 or Rank==18:
        return "9", CompareList[Rank]
    elif Rank==14:
        return "1", CompareList[Rank]
    elif Rank==16:
        return "8", CompareList[Rank]
    elif Rank==17:
        return "5", CompareList[Rank]
    else:
        return -1, -1


# Define function to recognize letter:
def RecognizeLetter(InputPath):
    # Letter database path:
    ImageAbstract="D:\\DATABASE\\CurrencyCharacter\\"
    
    # Comparing input image with database image:
    CompareList=[]
    for i in range(10, 36):
        AbstractPath=ImageAbstract+str(i)+".jpg"
        Std=Compare(InputPath, AbstractPath, str(i), 1)
        CompareList.append(Std)
    for i in range(200, 204):
        AbstractPath=ImageAbstract+str(i)+".jpg"
        Std=Compare(InputPath, AbstractPath, str(i), 1)
        CompareList.append(Std)
    
    # Find the most similar one and return:
    ListMin=CompareList.copy()
    list.sort(ListMin)
    Rank=CompareList.index(ListMin[0])
    if Rank==14:
        Rank=CompareList.index(ListMin[1])
    for i in range(0, 26):
        if Rank==i:
            return chr(i+65), CompareList[i]
    if Rank==26 or Rank==27:
        return "J", CompareList[Rank]
    elif Rank==28:
        return "I", CompareList[Rank]
    elif Rank==29:
        return "C", CompareList[Rank]
    else:
        return -1, -1


# Define funtion to define recognize character as number or letter:
def RecognizeCharacter(OriginInputPath, Length, LetterList, SerialCharacterList):
    for i in range(0, Length):
        SerialCharacterPath=OriginInputPath+str(i)+".jpg"
        k=0
        for j in range(0, len(LetterList)):
            if i==LetterList[j]:
                k=1
        if k==1:
            Letter, TruthValue=RecognizeLetter(SerialCharacterPath)
            SerialCharacterList.append(Letter)
        else:
            Number, TruthValue=RecognizeNumber(SerialCharacterPath)
            SerialCharacterList.append(Number)
    return SerialCharacterList
    

# Define funtion to recognize currency's serial number:
def RecognizeSerialNumber(InputPath, CurrencyType, Facevalue):
    # Path needed to store temporary image:
    ImageNote="D:\\DATABASE\\Note.jpg"
    ImageNoteThresh="D:\\DATABASE\\NoteThresh.jpg"
    ImageOriginSerialCharacter="D:\\DATABASE\\Element"
    
    # Load image:
    Img=cv2.imread(InputPath)
    H, W=Img.shape[:2]
    
    # List needed to store temporary information:
    Set=[]
    SetSeries=[]
    SetInX=[]
    SerialCharacterList=[]
    
    # Serial number recognition for EUR:
    if CurrencyType==0:
        # Path needed to store temporary image:
        NoteColorFilter="D:\\DATABASE\\NoteColorFilter.jpg"
        NoteSubColor="D:\\DATABASE\\NoteSubColor.jpg"
        
        ColorFilter(InputPath, NoteColorFilter, NoteSubColor, 80)
        
        ExtractDetail(NoteSubColor, Set, 5, 5, 50, 50/H, 20/H, 50/W, 1/W, 1, 1/10)
        SearchSeries(ImageNote, Set, SetSeries, 0, 10, 15, 6)
        
        # Decide its legality:
        if len(SetSeries)!=12:
            return -1
        
        # Recognize serial number character:
        RearrangeListInX(SetSeries, SetInX)
        NormalizeElement(ImageNoteThresh, SetInX)
        for i in range(0, len(SetInX)):
            SerialCharacterPath=ImageOriginSerialCharacter+str(i)+".jpg"
            Number, NumberTruthValue=RecognizeNumber(SerialCharacterPath)
            if i==0 or i==1:
                Letter, LetterTruthValue=RecognizeLetter(SerialCharacterPath)
                if NumberTruthValue<LetterTruthValue or Letter=="Q":
                    SerialCharacterList.append(Number)
                else:
                    SerialCharacterList.append(Letter)
            else:
                SerialCharacterList.append(Number)
        return SerialCharacterList
    
    # Serial number recognition for GBP-NEW:
    if CurrencyType==1:
        # Path needed to store temporary image:
        NoteNumberROI="D:\\DATABASE\\NoteNumberROI.jpg"
        
        # Cut ROI area:
        H, W=Img.shape[:2]
        Img[0:H, 0:int(W*(2/3))]=255
        cv2.imwrite(NoteNumberROI, Img)
        
        ExtractDetail(NoteNumberROI, Set, 5, 5, 50, 50/H, 20/H, 40/W, 2/W, 1, 1/10)
        SearchSeries(ImageNote, Set, SetSeries, 0, 20, 15, 6)
        
        # Decide its legality:
        if len(SetSeries)!=10:
            return -1
        
        # Recognize serial number character:
        RearrangeListInX(SetSeries, SetInX)
        NormalizeElement(ImageNoteThresh, SetInX)
        LetterList=[0, 1]
        RecognizeCharacter(ImageOriginSerialCharacter, len(SetInX), LetterList, SerialCharacterList)
        return SerialCharacterList
    
    # Serial number recognition for GBP-OLD:
    if CurrencyType==2:
        # Path needed to store temporary image:
        NoteNumberROI="D:\\DATABASE\\NoteNumberROI.jpg"
        
        # Cut ROI area:
        Img=Img[0:H, int(W*(4/5)):W]
        Img=cv2.transpose(Img, Img)
        Img=cv2.flip(Img, 0)
        cv2.imwrite(NoteNumberROI, Img)
        
        H, W=Img.shape[:2]
        ExtractDetail(NoteNumberROI, Set, 5, 5, 50, 50/H, 25/H, 40/W, 2/W, 1, 1/10)
        SearchSeries(NoteNumberROI, Set, SetSeries, 0, 20, 15, 6)
        
        # Decide its legality:
        if len(SetSeries)!=10:
            return -1
        
        # Recognize serial number character:
        RearrangeListInX(SetSeries, SetInX)
        NormalizeElement(ImageNoteThresh, SetInX)
        LetterList=[0, 1]
        RecognizeCharacter(ImageOriginSerialCharacter, len(SetInX), LetterList, SerialCharacterList)
        return SerialCharacterList
    
    # Serial number recognition for HKD:
    if CurrencyType==3 or CurrencyType==4:
        # Path needed to store temporary image:
        NoteColorFilter="D:\\DATABASE\\NoteColorFilter.jpg"
        NoteSubColor="D:\\DATABASE\\NoteSubColor.jpg"
        NoteSubColorROI="D:\\DATABASE\\NoteSubColorROI.jpg"
        
        ColorFilter(InputPath, NoteColorFilter, NoteSubColor)
        
        ExtractDetail(NoteSubColor, Set, 1, 1, 70, 80/H, 20/H, 50/W, 2/W, 1, 1/10)
        SearchSeries(ImageNote, Set, SetSeries, 0, 10, 10, 3)
        
        # Decide its legality and find another area to search serial number:
        if len(SetSeries)!=8:
            Set=[]
            Img=cv2.imread(NoteSubColor, cv2.IMREAD_GRAYSCALE)
            H, W=Img.shape[:2]
            Img[int(H*(1/4)):H, 0:W]=255
            cv2.imwrite(NoteSubColorROI, Img)
            ExtractDetail(NoteSubColorROI, Set, 1, 1, 70, 50/H, 20/H, 50/W, 2/W, 1, 1/10)
            SearchSeries(ImageNote, Set, SetSeries, 0, 10, 10, 3)
            if len(SetSeries)!=8:
                return -1
        
        # Recognize serial number character:
        RearrangeListInX(SetSeries, SetInX)
        NormalizeElement(ImageNoteThresh, SetInX)
        LetterList=[0, 1]
        RecognizeCharacter(ImageOriginSerialCharacter, len(SetInX), LetterList, SerialCharacterList)
        return SerialCharacterList
    
    # Serial number recognition for JPY:
    if CurrencyType==5:
        # Path needed to store temporary image:
        NoteColorFilter="D:\\DATABASE\\NoteColorFilter.jpg"
        NoteSubColor="D:\\DATABASE\\NoteSubColor.jpg"
        NoteSubColorROI="D:\\DATABASE\\NoteSubColorROI.jpg"
        
        ColorFilter(InputPath, NoteColorFilter, NoteSubColor)
        
        # Cut ROI area:
        Img=cv2.imread(NoteSubColor, cv2.IMREAD_GRAYSCALE)
        H, W=Img.shape[:2]
        Img[0:H, int(W*(1/2)):W]=255
        cv2.imwrite(NoteSubColorROI, Img)
        
        ExtractDetail(NoteSubColorROI, Set, 1, 1, 70, 60/H, 20/H, 50/W, 2/W, 1, 1/10)
        SearchSeries(ImageNote, Set, SetSeries, 0, 10, 15, 6)
        
        # Decide its legality:
        if len(SetSeries)!=9:
            return -1
        
        # Recognize serial number character:
        RearrangeListInX(SetSeries, SetInX)
        NormalizeElement(ImageNoteThresh, SetInX)
        LetterList=[0, 1, 8]
        RecognizeCharacter(ImageOriginSerialCharacter, len(SetInX), LetterList, SerialCharacterList)
        return SerialCharacterList
    
    # Serial number recognition for KRW:
    if CurrencyType==6:
        # Path needed to store temporary image:
        NoteNumberROI="D:\\DATABASE\\NoteNumberROI.jpg"
        
        # Cut ROI area:
        H, W=Img.shape[:2]
        Img[0:H, int(W*(1/2)):W]=255
        cv2.imwrite(NoteNumberROI, Img)
        
        ExtractDetail(NoteNumberROI, Set, 5, 5, 50, 50/H, 30/H, 30/W, 10/W, 1, 1/5)
        SearchSeries(ImageNote, Set, SetSeries, 0, 15, 15, 6)
        
        # Decide its legality:
        if len(SetSeries)!=10:
            return -1
        
        # Recognize serial number character:
        RearrangeListInX(SetSeries, SetInX)
        NormalizeElement(ImageNoteThresh, SetInX)
        LetterList=[0, 1, 9]
        RecognizeCharacter(ImageOriginSerialCharacter, len(SetInX), LetterList, SerialCharacterList)
        return SerialCharacterList
    
    # Serial number recognition for RMB:
    if CurrencyType==7:
        # List needed to store temporary information:
        SetInY=[]
        
        ExtractDetail(ImageNote, Set, 5, 5, 30, 50/H, 30/H, 30/W, 3/W, 1, 1/10)
        SearchSeries(ImageNote, Set, SetSeries, 0, 15, 15, 6)
        
        # Decide its legality:
        if len(SetSeries)!=10:
            return -1
        
        # Recognize serial number character for small face value:
        RearrangeListInX(SetSeries, SetInX)
        NormalizeElement(ImageNoteThresh, SetInX)
        if Facevalue!=50 and Facevalue!=100:
            LetterList=[]
            RearrangeListInX(SetSeries, SetInY, 1)
            for i in range(0, len(SetInX)):
                if SetInX[i].Location[1]==SetInY[0].Location[1]:
                    LetterList.append(i)
                if SetInX[i].Location[1]==SetInY[1].Location[1]:
                    LetterList.append(i)
            RecognizeCharacter(ImageOriginSerialCharacter, len(SetInX), LetterList, SerialCharacterList)
            return SerialCharacterList
        
        # List needed to store temporary information:
        NI=[]
        NT=[]
        NN=[]
        LI=[]
        LT=[]
        LN=[]
        
        # Recognize serial number character for large face value:
        for i in range(0, len(SetInX)):
            SerialCharacterPath=ImageOriginSerialCharacter+str(i)+".jpg"
            Number, NumberTruthValue=RecognizeNumber(SerialCharacterPath)
            Letter, LetterTruthValue=RecognizeLetter(SerialCharacterPath)
            if NumberTruthValue<LetterTruthValue and i!=0:
                SerialCharacterList.append(Number)
                NI.append(i)
                NT.append(LetterTruthValue)
                NN.append(Letter)
            else:
                SerialCharacterList.append(Letter)
                if i!=0:
                    LI.append(i)
                    LT.append(LetterTruthValue)
                    LN.append(Number)
        
        # Process when encounter error:
        if len(LI)>1:
            LTMin=LT.copy()
            list.sort(LTMin)
            Rank0=LT.index(LTMin[0])
            for i in range(0, len(LT)):
                if i!=Rank0:
                    SerialCharacterList[LI[i]]=LN[i]
        if len(NI)>8:
            NTMin=NT.copy()
            list.sort(NTMin)
            Rank8=NT.index(NTMin[8])
            for i in range(0, len(NT)):
                if i==Rank8:
                    SerialCharacterList[NI[i]]=NN[i]
        
        # Return ultimate serial number:
        return SerialCharacterList
    
    # Serial number recognition for TWD:
    if CurrencyType==8:
        # Path needed to store temporary image:
        NoteColorFilter="D:\\DATABASE\\NoteColorFilter.jpg"
        
        # Find dominant color:
        DeviateB, DeviateG, DeviateR=ExtractColorDetail(InputPath)
        BGRList=[DeviateB, DeviateG, DeviateR]
        BGRListMin=BGRList.copy()
        list.sort(BGRListMin)
        RankMax=BGRList.index(BGRListMin[2])
        
        # Color filter of imgae:
        Img=cv2.imread(InputPath)
        channels=cv2.split(Img)
        i=0
        for channel in channels:
            if i==RankMax:
                channel[:]=0
            i=i+1
        cv2.merge(channels, Img)
        cv2.imwrite(NoteColorFilter, Img)
        
        # Load new image:
        Img=cv2.imread(NoteColorFilter, cv2.IMREAD_GRAYSCALE)
        Img=np.int16(Img.flatten())
        Average=np.average(Img)
        
        ExtractDetail(NoteColorFilter, Set, 3, 1, Average*0.85, 50/H, 30/H, 30/W, 10/W, 1, 1/5)
        SearchSeries(ImageNote, Set, SetSeries, 0, 10, 15, 6)
        
        # Decide its legality:
        if len(SetSeries)!=10:
            return -1
        
        # Recognize serial number character:
        RearrangeListInX(SetSeries, SetInX)
        NormalizeElement(ImageNoteThresh, SetInX)
        LetterList=[0, 1, 8, 9]
        RecognizeCharacter(ImageOriginSerialCharacter, len(SetInX), LetterList, SerialCharacterList)
        return SerialCharacterList
      
    # Serial number recognition for USD:
    if CurrencyType==9:
        ExtractDetail(ImageNote, Set, 5, 5, 50, 50/H, 30/H, 50/W, 10/W, 1, 1/3)
        SearchSeries(ImageNote, Set, SetSeries, 0, 10, 15, 6)
        
        # Decide its legality:
        if len(SetSeries)!=9 and len(SetSeries)!=10 and len(SetSeries)!=11:
            return -1
        
        # Recognize serial number character:
        RearrangeListInX(SetSeries, SetInX)
        NormalizeElement(ImageNoteThresh, SetInX)
        if len(SetInX)==9:
            LetterList=[0]
        elif len(SetInX)==10:
            LetterList=[0, 9]
        else:
            LetterList=[0, 1, 10]
        
        # Decide whether it is 'star note':
        RecognizeCharacter(ImageOriginSerialCharacter, len(SetInX), LetterList, SerialCharacterList)
        if len(SetInX)==9:
            SerialCharacterList.append("*")
        elif len(SetInX)==10:
            Number1, NumberTruthValue1=RecognizeNumber("D:\\DATABASE\\Element1.jpg")
            Letter1, LetterTruthValue1=RecognizeLetter("D:\\DATABASE\\Element1.jpg")
            Number9, NumberTruthValue9=RecognizeNumber("D:\\DATABASE\\Element9.jpg")
            Letter9, LetterTruthValue9=RecognizeLetter("D:\\DATABASE\\Element9.jpg")
            if LetterTruthValue1<NumberTruthValue1 and LetterTruthValue9>NumberTruthValue9:
                SerialCharacterList[1]=Letter1
                SerialCharacterList[9]=Number9
                SerialCharacterList.append("*")
        
        # Return ultimate serial number:
        return SerialCharacterList

######################---UPDATE RELATED FUNCTIONS BELOW:---#############################################################
# Define function to generate a normalized image from original one:
def AddRealCategory(InputPath, OutputPath, Ruler):
    # Path needed to store temporary image:
    ImageNote="D:\\DATABASE\\Note.jpg"
    
    # Use functions defined above to generate normalized image:
    IdentifyNote(InputPath, ImageNote, Ruler)
    Normalize(ImageNote, OutputPath)
    print("Category added successfully!")


# Define function to calculate average image from input:
def AddAverageAbstractCategory(FirstInputPath, i, j, OutputPath):
    Img=cv2.imread(FirstInputPath+"\\"+str(i)+".jpg")
    k=2
    for i in range(i+1, j+1):
        ImgNew=cv2.imread(FirstInputPath+"\\"+str(i)+".jpg")
        Img=Img*((k-1)/k)+ImgNew/k
        k=k+1
    cv2.imwrite(OutputPath, Img)


# Define funtion to plot color information with different color category:
def PlotDotsImage():
    # Indicate source path:
    SourcePathONE="D:\\DATABASE\\RMB\\1\\"
    SourcePathFIVE="D:\\DATABASE\\RMB\\5\\"
    SourcePathTEN="D:\\DATABASE\\RMB\\10\\"
    SourcePathTWE="D:\\DATABASE\\RMB\\20\\"
    SourcePathFIF="D:\\DATABASE\\RMB\\50\\"
    SourcePathHUN="D:\\DATABASE\\RMB\\100\\"
    ImageNote="D:\\DATABASE\\Note.jpg"
    
    # List needed to store temporary information:
    ONE=[]
    FIVE=[]
    TEN=[]
    TWE=[]
    FIF=[]
    HUN=[]
    ONEx=[]
    ONEy=[]
    ONEz=[]
    FIVEx=[]
    FIVEy=[]
    FIVEz=[]
    TENx=[]
    TENy=[]
    TENz=[]
    TWEx=[]
    TWEy=[]
    TWEz=[]
    FIFx=[]
    FIFy=[]
    FIFz=[]
    HUNx=[]
    HUNy=[]
    HUNz=[]
    OnePointsX=[]
    FivePointsX=[]
    TenPointsX=[]
    TwePointsX=[]
    FifPointsX=[]
    HunPointsX=[]
    OnePointsY=[]
    FivePointsY=[]
    TenPointsY=[]
    TwePointsY=[]
    FifPointsY=[]
    HunPointsY=[]
    
    # Calculate each category's color information
    for i in range(1, 6):
        Name="ONE"+str(i-1)
        Path=SourcePathONE+str(i)+".jpg"
        IdentifyNote(Path, ImageNote, 0)
        W=ExtractColorDetail(ImageNote, Name, ONE)
        ONEx.append(W[0])
        ONEy.append(W[1])
        ONEz.append(W[2])
        X, Y=ConvertCoordinate(W[0], W[1], W[2])
        OnePointsX.append(X)
        OnePointsY.append(Y)
        print(X, Y)
    print("\n")
    
    for i in range(1, 8):
        Name="FIVE"+str(i-1)
        Path=SourcePathFIVE+str(i)+".jpg"
        IdentifyNote(Path, ImageNote, 0)
        W=ExtractColorDetail(ImageNote, Name, FIVE)
        FIVEx.append(W[0])
        FIVEy.append(W[1])
        FIVEz.append(W[2])
        X, Y=ConvertCoordinate(W[0], W[1], W[2])
        FivePointsX.append(X)
        FivePointsY.append(Y)
        print(X, Y)
    print("\n")
    
    for i in range(1, 10):
        Name="TEN"+str(i-1)
        Path=SourcePathTEN+str(i)+".jpg"
        IdentifyNote(Path, ImageNote, 0)
        W=ExtractColorDetail(ImageNote, Name, TEN)
        TENx.append(W[0])
        TENy.append(W[1])
        TENz.append(W[2])
        X, Y=ConvertCoordinate(W[0], W[1], W[2])
        TenPointsX.append(X)
        TenPointsY.append(Y)
        print(X, Y)
    print("\n")
    
    for i in range(1, 7):
        Name="TWE"+str(i-1)
        Path=SourcePathTWE+str(i)+".jpg"
        IdentifyNote(Path, ImageNote, 0)
        W=ExtractColorDetail(ImageNote, Name, TWE)
        TWEx.append(W[0])
        TWEy.append(W[1])
        TWEz.append(W[2])
        X, Y=ConvertCoordinate(W[0], W[1], W[2])
        TwePointsX.append(X)
        TwePointsY.append(Y)
        print(X, Y)
    print("\n")
     
    for i in range(1, 6):
        Name="FIF"+str(i-1)
        Path=SourcePathFIF+str(i)+".jpg"
        IdentifyNote(Path, ImageNote, 0)
        W=ExtractColorDetail(ImageNote, Name, FIF)
        FIFx.append(W[0])
        FIFy.append(W[1])
        FIFz.append(W[2])
        X, Y=ConvertCoordinate(W[0], W[1], W[2])
        FifPointsX.append(X)
        FifPointsY.append(Y)
        print(X, Y)
    print("\n")
    
    for i in range(1, 8):
        Name="HUN"+str(i-1)
        Path=SourcePathHUN+str(i)+".jpg"
        IdentifyNote(Path, ImageNote, 0)
        W=ExtractColorDetail(ImageNote, Name, HUN)
        HUNx.append(W[0])
        HUNy.append(W[1])
        HUNz.append(W[2])
        X, Y=ConvertCoordinate(W[0], W[1], W[2])
        HunPointsX.append(X)
        HunPointsY.append(Y)
        print(X, Y)
    print("\n")
    
    # Draw 3-dimension image:
    fig=plt.figure()
    ax=Axes3D(fig)
    ax.scatter(ONEx, ONEy, ONEz, c='r')
    ax.scatter(FIVEx, FIVEy, FIVEz, c='b')
    ax.scatter(TENx, TENy, TENz, c='g')
    ax.scatter(TWEx, TWEy, TWEz, c='c')
    ax.scatter(FIFx, FIFy, FIFz, c='y')
    ax.scatter(HUNx, HUNy, HUNz, c='k')
    ax.set_xlabel('B')
    ax.set_ylabel('G')
    ax.set_zlabel('R')
    ax.view_init(elev=10, azim=30)
    plt.show()
    
    # Draw 2-dimension image:
    pl.scatter(OnePointsX, OnePointsY, s=5, c=u'r', marker=u'*')
    pl.scatter(FivePointsX, FivePointsY, s=5, c=u'b', marker=u'*')
    pl.scatter(TenPointsX, TenPointsY, s=5, c=u'g', marker=u'*')
    pl.scatter(TwePointsX, TwePointsY, s=5, c=u'c', marker=u'*')
    pl.scatter(FifPointsX, FifPointsY, s=5, c=u'y', marker=u'*')
    pl.scatter(HunPointsX, HunPointsY, s=5, c=u'k', marker=u'*')
    pl.show()