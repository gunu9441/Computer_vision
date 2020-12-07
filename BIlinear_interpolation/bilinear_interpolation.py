from PIL import Image

def bilinear(image, coord):
    pixel=image.load()
  
    if(coord[0]<0) : coord=(0,coord[1])
    if(coord[0]>image.size[0]-2) : coord=(image.size[0]-2,coord[1])
    if(coord[1]<0) : coord=(coord[0],0)
    if(coord[1]>image.size[1]-2) : coord=(coord[0],image.size[1]-2)
    
    if(coord[0]==int(coord[0]) and coord[1]==int(coord[1])) : 
        return pixel[coord]

    left = int(coord[0])
    right = int(coord[0])+1
    top = int(coord[1])
    bottom = int(coord[1])+1

    A = pixel[left, top]
    B = pixel[right, top]
    C = pixel[left, bottom]
    D = pixel[right, bottom]
    
    a=coord[0]-int(coord[0])
    b=coord[1]-int(coord[1])


    E=( A[0]+a*(B[0]-A[0]) , A[1]+a*(B[1]-A[1]) , A[2]+a*(B[2]-A[2]) )
    F=( C[0]+a*(D[0]-C[0]) , C[1]+a*(D[1]-C[1]) , C[2]+a*(D[2]-C[2]) )
    X=( int(E[0]+b*(F[0]-E[0])) , int(E[1]+b*(F[1]-E[1])) , int(E[2]+b*(F[2]-E[2])) )
    return X

def magnify(image, rate):
    newsize = (int(image.size[0]*rate), int(image.size[1]*rate))
    newimg = Image.new("RGB", newsize)
    newpixel = newimg.load()
    for i in range(newsize[0]):
        for j in range(newsize[1]):
            coord=(i/rate, j/rate)
            newpixel[i, j] = bilinear(image,coord)
        print(i)

    return newimg
    
im=Image.open("lena.jpg")
im2=magnify(im,2)
im2.save("lena_interpolation.jpg")
