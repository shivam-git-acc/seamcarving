#include<iostream>
#include<opencv2/opencv.hpp>
typedef unsigned int ui;
using namespace std;
//Transposing the matrix so that same function can be applied again
void Transpose(int height,int width,ui ***RGB,ui ***RGB_T)
{
     for(int i=0;i<height;i++)
    {
        for(int j=0;j<width;j++)
        {
            RGB_T[0][j][i]=RGB[0][i][j];
            RGB_T[1][j][i]=RGB[1][i][j];
            RGB_T[2][j][i]=RGB[2][i][j];
        }
    }
    
}
//backtracking to fetch the pixels that are supposed to be removed
void backtracking(ui ***RGB,char **backtrack,int posi,int posj,int *col_array)
{

    if(posi<0)
    {
        return;
    }
    if(backtrack[posi][posj]=='U')
    {
        col_array[posi]=posj; 
        RGB[0][posi][posj]=255;
        RGB[1][posi][posj]=0;
        RGB[2][posi][posj]=0;

        backtracking(RGB,backtrack,posi-1,posj,col_array);
    }
    else if(backtrack[posi][posj]=='L')
    {
        col_array[posi]=posj-1; 
        RGB[0][posi][posj-1]=255;
        RGB[1][posi][posj-1]=0;
        RGB[2][posi][posj-1]=0;
        backtracking(RGB,backtrack,posi-1,posj-1,col_array);
    }
        else if(backtrack[posi][posj]=='R')
    {
        col_array[posi]=posj+1; 
        RGB[0][posi][posj+1]=255;
        RGB[1][posi][posj+1]=0;
        RGB[2][posi][posj+1]=0;
        backtracking(RGB,backtrack,posi-1,posj+1,col_array);
    }
    return;


}
//dp to find min falling path sum and also create arr array to register the moves so that it can be backtracked
double MinSum(double **matrix,double **dp,int R,int C,int posi,int posj,char ** arr)
{
    if(dp[posi][posj]!=INT_MAX)
    {
        return dp[posi][posj];
    }

    if(posj==0)
    {
        double x=MinSum(matrix,dp,R,C,posi-1,posj,arr)+matrix[posi][posj];
        double y=MinSum(matrix,dp,R,C,posi-1,posj+1,arr)+matrix[posi][posj];
        if(x>y)
        {
            dp[posi][posj]=y;
            arr[posi-1][posj]='R';
            
        }
        else{
            dp[posi][posj]=x;
            arr[posi-1][posj]='U';
        }
        
        return dp[posi][posj];
    }

    else if(posj==C-1)
    {
        double x=MinSum(matrix,dp,R,C,posi-1,posj,arr)+matrix[posi][posj];
        double y=MinSum(matrix,dp,R,C,posi-1,posj-1,arr)+matrix[posi][posj];

        if(x>y)
        {
            dp[posi][posj]=y;
            arr[posi-1][posj]='L';
        }
        else{
            dp[posi][posj]=x;
            arr[posi-1][posj]='U';
        }

        return dp[posi][posj];                  
    }
    else
    {
        double x=MinSum(matrix,dp,R,C,posi-1,posj,arr)+matrix[posi][posj];
        double y=MinSum(matrix,dp,R,C,posi-1,posj+1,arr)+matrix[posi][posj];
        double z=MinSum(matrix,dp,R,C,posi-1,posj-1,arr)+matrix[posi][posj];

        if(x>y)
        {
            dp[posi][posj]=y;
            arr[posi-1][posj]='R';

        }
        else{
            dp[posi][posj]=x;
            arr[posi-1][posj]='U';
        }

        if( dp[posi][posj]>z)
        {
            dp[posi][posj]=z;
            arr[posi-1][posj]='L';

        }
        
        return dp[posi][posj];

    }

}
//modulus function that can deal with positive and negative
int mod(int a,int b)
{
 
    return (a+b)%b;
 
}

double calculate_gradient(int R,int G,int B)
{   
    return pow(R,2)+pow(G,2)+pow(B,2);

}
//dynamic allocation of RGB array
ui ***create_RGB_array(int channels,int height,int width)
{
     ui ***RGB=new ui**[channels];

     for(int i=0;i<channels;i++)
    {
        RGB[i]=new ui*[height];
        
        for(int j=0;j<height;j++)
        {   
           
            RGB[i][j]=new ui[width];
        }
    }

    return RGB;
        

}
//extracting data from Image in RGB channel
void extract_RGB_data(ui ***RGB,cv::Mat &image,int height,int width)
{
    for(int i=0;i<height;i++)
    {
        for(int j=0;j<width;j++)
        {
         
            cv::Vec3b pixel = image.at<cv::Vec3b>(i,j);
            //Red is pixel[2]
            RGB[0][i][j]=pixel[2];
            //Green
            RGB[1][i][j]=pixel[1];
            //Blue is pixel[0]
            RGB[2][i][j]=pixel[0];

        }
    }
}
//creating energy matrix for identifying uniform patches
double **create_energy_matrix(int height,int width,ui ***RGB)
{
    double **Energy=new double*[height];
    bool flag=false;
    for(int i=0;i<height;i++)
    {
        Energy[i]=new double[width];
    }

        for(int i=0;i<height;i++)
    {   flag=false;
        for(int j=0;j<width;j++)
        {
            int delta_R_in_y=RGB[0][mod(i-1,height)][j]-RGB[0][mod(i+1,height)][j];
            int delta_G_in_y=RGB[1][mod(i-1,height)][j]-RGB[1][mod(i+1,height)][j];
            int delta_B_in_y=RGB[2][mod(i-1,height)][j]-RGB[2][mod(i+1,height)][j];
            int delta_R_in_x=RGB[0][i][mod(j-1,width)]-RGB[0][i][mod(j+1,width)];
            int delta_G_in_x=RGB[1][i][mod(j-1,width)]-RGB[1][i][mod(j+1,width)];
            int delta_B_in_x=RGB[2][i][mod(j-1,width)]-RGB[2][i][mod(j+1,width)];


            
            Energy[i][j]=sqrt(calculate_gradient(delta_R_in_x,delta_G_in_x,delta_B_in_x) + calculate_gradient(delta_R_in_y,delta_G_in_y,delta_B_in_y) );               
         

        }
      
     
    }
    return Energy;
}
//create dp matrix
double **calculate_dp(int height,int width)
{
    double **dp=new double*[height];

     for(int i=0;i<height;i++)
    {
        dp[i]=new double[width];
        for(int j=0;j<width;j++)
        {
            dp[i][j]=INT_MAX;
        }
        
    }
    
    return dp;

}
//creating backtrack matrix
char **create_backtrack(int height,int width)
{
    char **backtrack=new char*[height-1];
    for(int i=0;i<height-1;i++)
    {
        backtrack[i]=new char[width];
        for(int j=0;j<width;j++)
        {
            backtrack[i][j]='\0';
        }
        
    }
    return backtrack;
}
//calling dp to find seam
int find_seam(int R,int C,int posi,char **backtrack,double **Energy,double** dp)
{
    double sum,minimum=INT_MAX;
    int col,posj;
    for(int j=0;j<C;j++)
    {
        
        posj=j;
  
        sum=MinSum(Energy,dp,R,C,posi,posj,backtrack);
  
        if(minimum>sum)
        {
            minimum=sum;
            col=j;
   

        }
    
    }

    return col;
}
//taking the filteredv RGB data 
void copy_in_new(int R,int C,ui ***RGB,ui***newRGB,int *col_array)
{
    int newi,newj;
    for(int i=0,newi=0;i<R;i++,newi++)
    {
        for(int j=0,newj=0;j<=C;j++)
        {
            if(j!=col_array[i])
            {
                newRGB[0][newi][newj]=RGB[0][i][j];
                newRGB[1][newi][newj]=RGB[1][i][j];
                newRGB[2][newi][newj]=RGB[2][i][j];
                
                newj++;
            }
        }

    }
}
//coverting array to image depending on row or column transformation
void array_to_image(int R,int C,ui ***RGB,cv::Mat &img,bool flag)
{
    if(flag==true)
    {
        for(int i=0;i<R;i++)
        {
            for(int j=0;j<C;j++)
            {
                cv::Vec3b &pixel = img.at<cv::Vec3b>(i, j);
                pixel[0]=RGB[2][i][j];
                pixel[1]=RGB[1][i][j];
                pixel[2]=RGB[0][i][j];
            }

        }
    }
    else{
        for(int i=0;i<R;i++)
        {
            for(int j=0;j<C;j++)
            {
                cv::Vec3b &pixel = img.at<cv::Vec3b>(j, i);
                pixel[0]=RGB[2][i][j];
                pixel[1]=RGB[1][i][j];
                pixel[2]=RGB[0][i][j];
            }

        }
    }

    
}
//deallocating the heap
void destruct(double **Energy,char **backtrack,double **dp,ui ***RGB,int height,int width,int channels)
{
    for (int i=0; i<height;i++)
    {
         delete[] Energy[i];   
    }
    delete[] Energy; 

    for (int i=0;i<height-1;i++)
    {
         delete[] backtrack[i];   
    }
    delete[] backtrack; 

    for (int i=0; i<height;i++)
    {
         delete[] dp[i];   
    }
    delete[] dp; 

    for (int i=0;i<channels;i++) {
        for (int j=0;j<height;j++) {
            delete[] RGB[i][j];
        }
        delete[] RGB[i];
    }
    delete[] RGB;
}
//module for col compression
ui *** compress_in_cols(int channels,int height,int width,ui***RGB)
{

    
    double **Energy=create_energy_matrix(height,width,RGB);
    double **dp=calculate_dp(height,width);
    char **backtrack=create_backtrack(height,width);

    int R=height;
    int C=width;

    for(int i=0;i<C;i++)
    {
            dp[0][i]=Energy[0][i];
            
        
    }
    int posi=R-1,posj;
    double sum,minimum=INT_MAX;
    int col;

    col=find_seam(R,C,posi,backtrack,Energy,dp);

    int *col_array=new int[R];

    RGB[0][R-1][col]=255;
    RGB[1][R-1][col]=0;
    RGB[2][R-1][col]=0;

    col_array[R-1]=col;

   

    backtracking(RGB,backtrack,R-2,col,col_array);

    C--;

    ui ***newRGB=create_RGB_array(channels,R,C);

    copy_in_new(R,C,RGB,newRGB,col_array);

    cv::Mat img(height, width, CV_8UC3);


   for(int i=0;i<R;i++)
        {
            for(int j=0;j<C;j++)
            {
                cv::Vec3b &pixel = img.at<cv::Vec3b>(i, j);
                pixel[0]=RGB[2][i][j];
                pixel[1]=RGB[1][i][j];
                pixel[2]=RGB[0][i][j];
            }

        }

        


    cv::imshow("Image from Modified RGB Array", img);
    cv::waitKey(1);
  
    destruct(Energy,backtrack,dp,RGB,height,width,channels);
    
    
    RGB=newRGB;

    return RGB;
}
//module for compression in rows
ui*** compress_in_rows(int channels,int height,int width,ui ***RGB)
{
    double **Energy=create_energy_matrix(height,width,RGB);

    double **dp=calculate_dp(height,width);
    char **backtrack=create_backtrack(height,width);
    int R=height;
    int C=width;

    for(int i=0;i<C;i++)
    {
        dp[0][i]=Energy[0][i];
            
    }

    int posi=R-1,posj;
    double sum,minimum=INT_MAX;
    int col;
   
    col=find_seam(R,C,posi,backtrack,Energy,dp);

    int *col_array=new int[R];

    RGB[0][R-1][col]=255;
    RGB[1][R-1][col]=0;
    RGB[2][R-1][col]=0;

    col_array[R-1]=col;
    
  
    backtracking(RGB,backtrack,R-2,col,col_array);

    C--;

    ui ***newRGB=create_RGB_array(channels,R,C);
 

    copy_in_new(R,C,RGB,newRGB,col_array);
  
  

    cv::Mat img(width, height, CV_8UC3);


    array_to_image(R,C,RGB,img,false);

  
    cv::imshow("Image from Modified RGB Array", img);
    cv::waitKey(1);
  
    destruct(Energy,backtrack,dp,RGB,height,width,channels);
    RGB=newRGB;
  
    return RGB;

}


int main(int argc,char **argv)
{
    cv::Mat image=cv::imread(argv[1]);
    

    if (image.empty()) {
        std::cout << "Could not read the image" << std::endl;
        return 1;
    }
    else{
        cout<<"Image read succesfully"<<endl;
    }


    int height = image.rows;
    int width = image.cols;
    int channels=3;


    cout << "Image dimensions: " << height << "x" << width << std::endl;

    cout<<"creating 3d mat"<<endl;

    ui ***RGB=create_RGB_array(channels,height,width);
    ui ***RGB_T=create_RGB_array(channels,width,height);

    cout<<"Start extraction\n";
    fflush(stdout);

    extract_RGB_data(RGB,image,height,width);

    cout<<"RGB values extracted\n"<<endl;

    cout<<"creating energy matrix\n";

    cout<<"Enter how much rows you want to remove? "<<endl;
    int comp_in_rows;
    cin>>comp_in_rows;
    int comp_in_cols;
    cout<<"Enter how much cols you want to remove? "<<endl;
    cin>>comp_in_cols;
   
    if(comp_in_rows>=height || comp_in_cols>=width)
    {
        cout<<"compression not possible!!"<<endl;
        return 0;
    }

    cout<<"Compression in progress, please wait for few seconds"<<endl;
    while(comp_in_cols>0)
    {
        RGB=compress_in_cols(channels,height,width,RGB);
        width--;
        comp_in_cols--;
    }

    Transpose(height,width,RGB,RGB_T);

    while(comp_in_rows>0)
    {
        RGB_T=compress_in_rows(channels,width,height,RGB_T);
        height--;
        comp_in_rows--;
    }

    cv::Mat compressed_image(height, width, CV_8UC3);

  
    array_to_image(width,height,RGB_T,compressed_image,false);

    cout<<"Image compressed Succesfully!!!"<<endl;
        
    bool success = imwrite("output.jpg", compressed_image);

    if (!success) {
        cout<<"Error: Could not save the image!"<<endl;
    } else {
        cout<<"Image saved successfully!"<<endl;
    }

 

    return 0;
}