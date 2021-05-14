import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

class Animal_Shougi_Super:
    def __init__(self,):
        self.Bint=np.array([[-1,-1,-1],[0,-1,0],[0,1,0],[1,1,1]])#盤面：１がターンの人、-1が相手
        self.Bstr=np.array([["K","L","Z"],["#","H","#"],["#","H","#"],["Z","L","K"]])#盤面、コマ名
        self.m_now=np.array([["#","#"],["#","#"],["#","#"]])#ターンの人の墓地K,L,H
        self.m_next=np.array([["#","#"],["#","#"],["#","#"]])#墓地
        self.turn=0#ターン数
        self.first=1#先手が誰か
        self.first_change=0#ゲームがスタートしたら1になる
        self.start=0#ゲームスタートしたターン数
        self.friend=0#なんだっけ
        self.movable={"H":[[-1,0]],
                      "K":[[1,0],[0,1],[0,-1],[-1,0]],
                      "L":[[0,1],[-1,1],[1,1],[1,0],[-1,-1],[1,-1],[-1,0],[0,-1]],
                      "Z":[[1,-1],[1,1],[-1,-1],[-1,1]],
                      "N":[[-1,-1],[-1,0],[-1,1],[0,1],[0,-1],[1,0]]}#可動域
        #self.ad="./Downloads/木下vs坂部/"#写真保存するアドレス
        self.ad="./Downloads/Photos/"

    def create_cells(self,a,hori,verti):#格子点を作成、aが最低限2隅、horiが横のマス数、vertiが縦のマス数
        c=[]
        dx=(a[1][0]-a[0][0])//hori
        dy=(a[1][1]-a[0][1])//verti
        for i in range (0,hori+1):
            for j in range (0,verti+1):
                c.append([a[0][0]+dx*i,a[0][1]+dy*j])
        #print(len(c))
        return c
    def cell_decomposition(self,file):#file(写真)をマス"square"に分割する
        im=cv2.imread(self.ad+file)
        im_marked = im.copy()
        src_pts=[]
        #src_pts+=self.create_cells([[672,530],[1363,1052]],4,3)
        #src_pts+=self.create_cells([[644,46],[903,406]],2,3)
        #src_pts+=self.create_cells([[1069,34],[1329,389]],2,3)
        src_pts+=self.create_cells([[675,496],[1378,1021]],4,3)#盤面の最低限2すみ
        src_pts+=self.create_cells([[688,7],[932,371]],2,3)#墓地１（左）の最低限2すみ
        src_pts+=self.create_cells([[1102,18],[1373,377]],2,3)#墓地２（右）の最低限2すみ

        #マーカーを記したものも保存しておく
        for pt in src_pts:
            cv2.drawMarker(im_marked, tuple(pt), (0, 255, 0), thickness=4)
        cv2.imwrite(self.ad+file+"_marked.jpg", im_marked)
        #print(im[1:3][0])
        #print(src_pts[0][0])
        #print(src_pts)
        src_pts=np.array(src_pts, dtype=np.int32)

        #各マスの画像を生成, 1~24

        it=1
        for i in range (0,39):
            if i in [3,7,11,15,16,17,18,19,23,27,28,29,30,31,35]:
                continue
            image=im[src_pts[i][1]:src_pts[i+1][1],src_pts[i][0]:src_pts[i+5][0],:].copy()
            cv2.imwrite(self.ad+file+"_square"+str(it)+".jpg", image)
            it+=1

    def piece_likelihood(self,image,i):#写真の色分布を見てコマがあるかどうかの程度を出力

        img=cv2.imread(self.ad+image)
        n,m,z=img.shape
        #x=-(kurtosis(np.resize(img[10:n-10,10:m-10,2],(1,n*m))[0]))
        #print(x)
        hist0=cv2.calcHist([img[10:n-10,10:m-10,:]],[0],None,[256],[0,256])#B
        hist1=cv2.calcHist([img[10:n-10,10:m-10,:]],[1],None,[256],[0,256])#G
        hist2=cv2.calcHist([img[10:n-10,10:m-10,:]],[2],None,[256],[0,256])#R
        #plt.plot(hist0)
        #plt.plot(hist1)
        #plt.plot(hist2,color = "r")
        #print(np.argmax(hist0),np.argmax(hist1),np.argmax(hist2))
        #plt.show()

        #色成分が非負なところの数を計算する。ただしある程度閾値を設け、ピーク付近は計算しないようにする。

        n=len(hist0)
        s=0

        if i>=13:
            for i in range(0,n):
                if (hist0[i]>20) and (i<np.argmax(hist0)-10 or i>np.argmax(hist0)+10):
                    s+=1
                if (hist1[i]>20) and (i<np.argmax(hist1)-10 or i>np.argmax(hist1)+10):
                    s+=1
                if (hist2[i]>20) and (i<np.argmax(hist2)-10 or i>np.argmax(hist2)+10):
                    s+=1
        else:
            for i in range(0,n):
                if (hist0[i]>100) and (i<np.argmax(hist0)-10 or i>np.argmax(hist0)+10):
                    s+=1
                if (hist1[i]>100) and (i<np.argmax(hist1)-10 or i>np.argmax(hist1)+10):
                    s+=1
                if (hist2[i]>100) and (i<np.argmax(hist2)-10 or i>np.argmax(hist2)+10):
                    s+=1
        return s

    def piece_position(self,image):#あるターンのimageにコマがあるかないかを判断する
        A=np.zeros(24)
        for i in range (1,25):
            A[i-1]=self.piece_likelihood(image+"_square"+str(i)+".jpg",i)
            #print(i,A[i-1])

        #piece_likelihoodで大きい値を達成するもの8つを選ぶ、その箇所を3で記した行列を出力
        index=A.argsort()[-8:][::-1]
        for i in range (0,24):
            if i not in index:
                A[i]=0
            else:
                A[i]=3
        B=np.fliplr(np.transpose(A[0:12].reshape((3,4),order="F")))#盤面
        m1=A[12:18].reshape((3,2),order="F")#左墓地
        m2=A[18:25].reshape((3,2),order="F")#右墓地
        return B,m1,m2

    def color_likelihood(self,im1,im2):#二つの画像の色分布の距離を計算
        img1=cv2.imread(self.ad+im1)
        img2=cv2.imread(self.ad+im2)
        n,m,z=img1.shape
        s=0
        for i in range (0,3):#RGB全てに対してやる
            hist1=cv2.calcHist([img1[10:n-10,10:m-10,:]],[i],None,[256],[0,256])
            hist2=cv2.calcHist([img2[10:n-10,10:m-10,:]],[i],None,[256],[0,256])
            j=np.argmax(hist1)
            k=np.argmax(hist2)
            #print(j-k)
            #if |j-k|>19: return 100000000
            #ノイズが必ず入るのでピークは合わせる
            if j>k:
                diff=hist1[j-k:]-hist2[:256-(j-k)]
                s+=np.linalg.norm(diff[j-k:256-(j-k)])
            else:
                diff=hist1[k-j:]-hist2[:256-(k-j)]
                s+=np.linalg.norm(diff[k-j:256-(k-j)])
        return s

    def update(self,Bint,m1,m2):#更新則
        #まず開始時点を定める、最初はself.first_change=0である。

        if self.first_change==1: #ゲーム開始後
            if self.turn%2==self.start%2+self.first%2:#左の人のターンのとき

                B_before=self.Bint.copy()
                B_after=np.flipud(np.fliplr(Bint.copy()))#必ずしたをそのターンの人に向けるフリップする(左の時だけ特別な処置)
                B_change=B_before-B_after#盤面変化
                A_a=m2#ターンの人の墓地

            else:
                A_a=m1
                B_before=self.Bint.copy()
                B_after=Bint.copy()
                B_change=B_before-B_after

        elif self.first_change==0:#最初の変化が起きるまではこっち

            B_before=self.Bint.copy()
            B_after=Bint.copy()
            B_change=B_before-B_after

            if (-1 in B_change):#左が先手なら初めての変化は必ずこっち

                self.first=2#先手設定
                self.first_change=1#ゲーム開始
                self.start=self.turn#ゲーム開始ターン数
                B_after=np.flipud(np.fliplr(Bint.copy()))#必ずしたをそのターンの人に向けるフリップする
                B_change=B_before-B_after#盤面変化
                A_a=m2#そのターンの人の墓地
                #print(B_change)

            elif (1 in B_change):#右が先手なら初めての変化は必ずこっち
                A_a=m1
                self.first=1
                self.first_change=1
                self.start=self.turn
                B_after=Bint.copy()
                B_change=B_before-B_after

        #以下変化パターンで分ける

        if (-3 in B_change) and (1 in B_change):#コマがただ移動した時

            #print("in1")
            #B_change で-3が移動先、1が移動元

            #インデックスの特定
            i_a=np.where(B_change==-3)[0][0]
            j_a=np.where(B_change==-3)[1][0]
            i_b=np.where(B_change==1)[0][0]
            j_b=np.where(B_change==1)[1][0]

            #後は入れ替え
            self.Bint[i_a][j_a]=self.Bint[i_b][j_b].copy()
            self.Bstr[i_a][j_a]=self.Bstr[i_b][j_b].copy()

            self.Bint[i_b][j_b]=0
            self.Bstr[i_b][j_b]="#"

        elif (1 in B_change):#相手のコマをとるムーブの時（一番厄介）
            #print("in2")


            A_b=self.m_now #1ターン前の墓地の状況

            A_change=np.logical_xor((A_b!="#"),(A_a!=0))#墓地の変化
            #墓地の変化した場所を特定、それによってとられたコマの種類がわかる
            i=np.where(A_change==True)[0][0]
            j=np.where(A_change==True)[1][0]
            taken=np.array([["K","K"],["L","L"],["H","H"]])[i][j]

            #移動したコマを特定
            i_b=np.where(B_change==1)[0][0]
            j_b=np.where(B_change==1)[1][0]
            moved=self.Bstr[i_b][j_b]

            #取られたコマを参照して、移動したコマの移動先を特定する。
            #すなわち、移動したコマの可動域を見て取られたコマと同じ種類の相手のコマがあるか探索する（最大2、最小１）
            #possibilityに候補を格納

            move_to=self.movable[moved]
            possibility=[]
            #print(move_to,i_b,j_b)

            for k in move_to:

                if 0<=i_b+k[0]<=3 and 0<=j_b+k[1]<=2:
                    #print(self.Bint[i_b+k[0]][j_b+k[1]],self.Bstr[i_b+k[0]][j_b+k[1]])
                    #print(taken)
                    if self.Bint[i_b+k[0]][j_b+k[1]]==-1 and self.Bstr[i_b+k[0]][j_b+k[1]]==taken:
                        possibility.append([i_b+k[0],j_b+k[1]])


            #print(possibility)

            if len(possibility)==1:#行先が1通りしかない場合は、簡単

                self.Bint[i_b][j_b]=0
                self.Bint[possibility[0][0],possibility[0][1]]=1
                self.Bstr[possibility[0][0],possibility[0][1]]=self.Bstr[i_b][j_b].copy()
                self.Bstr[i_b][j_b]="#"

                self.m_now[i][j]=taken

            else:#行先が２通りの場合は色分布の比較が必要

                if self.turn%2==self.start%2+self.first%2:#左のターンの場合

                    p=[3,6,9,12,2,5,8,11,1,5,7,10]
                    q=[10,7,4,1,11,8,5,2,12,9,6,3]
                    #print(possibility[0][0]+possibility[0][1]*4)
                    #print(possibility[1][0]+possibility[1][1]*4)
                    index_im1=p[possibility[0][0]+possibility[0][1]*4]
                    index_im2=p[possibility[1][0]+possibility[1][1]*4]
                    index_model_im=q[i_b+j_b*4]
                    im1="image"+str(self.turn)+".png_square"+str(index_im1)+".jpg"#候補1の変化後の写真
                    im2="image"+str(self.turn)+".png_square"+str(index_im2)+".jpg"#候補2の変化後の写真
                    model_im="image"+str(self.turn-1)+".png_square"+str(index_model_im)+".jpg"#移動したコマの移動前の写真


                    #色分布の距離を計算
                    like1=self.color_likelihood(im1,model_im)
                    like2=self.color_likelihood(im2,model_im)

                    if (like1>like2):

                        self.Bint[i_b][j_b]=0
                        self.Bint[possibility[0][0],possibility[0][1]]=1
                        self.Bstr[possibility[0][0],possibility[0][1]]=self.Bstr[i_b][j_b].copy()
                        self.Bstr[i_b][j_b]="#"
                        self.m_now[i][j]=taken
                    else:
                        self.Bint[i_b][j_b]=0
                        self.Bint[possibility[1][0],possibility[1][1]]=1
                        self.Bstr[possibility[1][0],possibility[1][1]]=self.Bstr[i_b][j_b].copy()
                        self.Bstr[i_b][j_b]="#"
                        self.m_now[i][j]=taken

                else:#右のターンのとき

                    q=[3,6,9,12,2,5,8,11,1,5,7,10]
                    p=[10,7,4,1,11,8,5,2,12,9,6,3]
                    #print(possibility)
                    #print(possibility[0][0]+possibility[0][1]*4)
                    #print(possibility[1][0]+possibility[1][1]*4)
                    index_im1=p[possibility[0][0]+possibility[0][1]*4]
                    index_im2=p[possibility[1][0]+possibility[1][1]*4]
                    index_model_im=q[i_b+j_b*4]
                    im1="image"+str(self.turn)+".png_square"+str(index_im1)+".jpg"
                    im2="image"+str(self.turn)+".png_square"+str(index_im2)+".jpg"
                    model_im="image"+str(self.turn-1)+".png_square"+str(index_model_im)+".jpg"

                    like1=self.color_likelihood(im1,model_im)
                    like2=self.color_likelihood(im2,model_im)

                    if (like1>like2):

                        self.Bint[i_b][j_b]=0
                        self.Bint[possibility[0][0],possibility[0][1]]=1
                        self.Bstr[possibility[0][0],possibility[0][1]]=self.Bstr[i_b][j_b].copy()
                        self.Bstr[i_b][j_b]="#"
                        self.m_now[i][j]=taken
                    else:
                        self.Bint[i_b][j_b]=0
                        self.Bint[possibility[1][0],possibility[1][1]]=1
                        self.Bstr[possibility[1][0],possibility[1][1]]=self.Bstr[i_b][j_b].copy()
                        self.Bstr[i_b][j_b]="#"
                        self.m_now[i][j]=taken
            #print("under construction")

        elif (-3 in B_change):#コマが墓地から召喚された時
            #print("in3")

            A_b=self.m_now
            A_change=np.logical_xor((A_b!="#"),(A_a!=0))#召喚されたコマを特定

            i_a=np.where(B_change==-3)[0][0]
            j_a=np.where(B_change==-3)[1][0]
            i_b=np.where(A_change==True)[0][0]
            j_b=np.where(A_change==True)[1][0]

            self.Bint[i_a][j_a]=1

            self.Bstr[i_a][j_a]=self.m_now[i_b][j_b].copy()
            self.m_now[i_b][j_b]="#"

        #print(self.Bint,self.Bstr)

        self.Bint=np.flipud(np.fliplr(-self.Bint))
        self.Bstr=np.flipud(np.fliplr(self.Bstr))
        tmp=self.m_now.copy()
        self.m_now=self.m_next.copy()
        self.m_next=tmp.copy()

    def step(self):#ターンの経過毎に発動

        self.turn+=1 #ターン数+1

        ##!save picture to "image{turn}.png" e.g. image11.jpg!##

        image_name="image"+str(self.turn)+".png"#写真の名前
        self.cell_decomposition(image_name)#写真をマス毎に分解
        Bint,m2,m1=self.piece_position(image_name)#ありなし判断

        #print(self.turn)
        #print(m2)
        #print(m1)
        #print(Bint)

        #img = mpimg.imread(self.ad+"image"+str(self.turn)+".png")
        #imgplot = plt.imshow(img)

        self.update(Bint,m1,m2)#盤面情報更新

        #plt.show()
