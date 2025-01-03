
#define CUDART_PI_F 3.141592654f


__device__ int get_index_gpu(int x, int y, int z,int ind_or,int ind_rad,int stride_x ,int stride_y, int stride_z, int stride_or,int stride_rad)
{
	return x*stride_y*stride_z*stride_or*stride_rad + y*stride_z*stride_or*stride_rad\
         + z*stride_or*stride_rad+ind_or*stride_rad+ind_rad;
}

__device__ void Get_Spatial_Pos(int* Ref_Coord, int* Vec_tr, float Radius, float* Direction)
{
    for(int k = 0;k < 3;++k)
    {
        Vec_tr[k] = float_as_int(Ref_Coord[k]+Radius*Direction[k]);
    }
}


__device__ void Spatial_Coordinates_Disk(int* Node_Coord, int* Node_Coord_new, float* radius, float* angle, float* Orth_Vec,
                                    int boundary_pos)
{
    int Vec_Tr_pos[3] = {0,0,0};

    Get_Spatial_Pos(&Node_Coord[0],&Vec_Tr_pos[0],radius[0],Orth_Vec);

    for(int k =0; k< 3; ++k)
    {
        Node_Coord_new[k] = Node_Coord[k];
    }

    int min_check = Vec_Tr_pos[0];

    int max_check = Vec_Tr_pos[0];

    for(int k = 1; k < 3;++k) 
    {
        if(min_check > Vec_Tr_pos[k]){min_check = Vec_Tr_pos[k];}

        if(max_check < Vec_Tr_pos[k]){max_check = Vec_Tr_pos[k];}
    }
    if(max_check <= boundary_pos-1 & min_check >= 0)
    {
        Node_Coord_new[0] = Vec_Tr_pos[0];

        Node_Coord_new[1] = Vec_Tr_pos[1];

        Node_Coord_new[2] = Vec_Tr_pos[2];
    }
}


__device__ float Spherical_Norm(float* func_1, float* func_2, int dim, float* Or_Array)
{
    float Theta = 0;

    float Integral = 0;

    for(int k = 0; k < int(dim);++k)
    {
        Theta = acosf(Or_Array[k*3+2]);

        Integral += (func_1[k]-func_2[k])*(func_1[k]-func_2[k])*sinf(Theta)/dim;
    }
    return Integral;
}

__device__ void argmin_sort(float* Vec,int* Vec_ind)
{

    float S_var = 0;
    int S_Ind = 0;

    for(int k = 0;k < 3; ++k)
    {
        S_var = fabsf(Vec[k]);
        S_Ind = Vec_ind[k];

        for(int l = k+1; l < 3;++l)
        {
            if(S_var > fabsf(Vec[l]))
            {
                S_var = fabsf(Vec[l]);
                S_Ind = Vec_ind[l];

                Vec[l] = Vec[k];
                Vec[k] = S_var;
                Vec_ind[l] = Vec_ind[k];
                Vec_ind[k] = S_Ind;
            }
        }
    }
}

__device__ void Norm_Null_Space(float* Input_Mat)
{

    float Norm = sqrtf(Input_Mat[0]*Input_Mat[0]+Input_Mat[3]*Input_Mat[3]+Input_Mat[6]*Input_Mat[6]);

    float Vec_1[3] = {Input_Mat[0]/Norm,Input_Mat[3]/Norm,Input_Mat[6]/Norm};   

    float Vec_2[3] = {0,0,0};

    float Vec_3[3] = {0,0,0};

    float Var_proj = 0;

    float Vec_Sort[3] = {Vec_1[0],Vec_1[1],Vec_1[2]};

    int Vec_ind[3] = {0,1,2};

    // Compute orthonormal Basis

    if(Vec_1[0] != 0 & Vec_1[1] != 0 & Vec_1[2] != 0)
    {
        argmin_sort(&Vec_Sort[0],&Vec_ind[0]);
        Var_proj = Vec_1[Vec_ind[0]];
        Vec_2[Vec_ind[0]] = 1-Var_proj*Vec_1[Vec_ind[0]];
        Vec_2[Vec_ind[1]] = -Var_proj*Vec_1[Vec_ind[1]];
        Vec_2[Vec_ind[2]] = -Var_proj*Vec_1[Vec_ind[2]];

        Norm = sqrtf(Vec_2[0]*Vec_2[0]+Vec_2[1]*Vec_2[1]+Vec_2[2]*Vec_2[2]);
        Vec_2[0] = Vec_2[0]/Norm;
        Vec_2[1] = Vec_2[1]/Norm;
        Vec_2[2] = Vec_2[2]/Norm;
        
        
        Vec_3[0] = Vec_2[2]*Vec_1[1]-Vec_2[1]*Vec_1[2];
        Vec_3[1] = Vec_2[0]*Vec_1[2]-Vec_2[2]*Vec_1[0];
        Vec_3[2] = Vec_2[1]*Vec_1[0]-Vec_2[0]*Vec_1[1];
        Norm = sqrtf(Vec_3[0]*Vec_3[0]+Vec_3[1]*Vec_3[1]+Vec_3[2]*Vec_3[2]);
        Vec_3[0] = Vec_3[0]/Norm;
        Vec_3[1] = Vec_3[1]/Norm;
        Vec_3[2] = Vec_3[2]/Norm;
    }

    else if(Vec_1[0] == 0)
     {
        Var_proj = Vec_1[0];
        Vec_2[0] = 1-Var_proj*Vec_1[0];
        Vec_2[1] = -Var_proj*Vec_1[1];
        Vec_2[2] = -Var_proj*Vec_1[2];
        Norm = sqrtf(Vec_2[0]*Vec_2[0]+Vec_2[1]*Vec_2[1]+Vec_2[2]*Vec_2[2]);
        Vec_2[0] = Vec_2[0]/Norm;
        Vec_2[1] = Vec_2[1]/Norm;
        Vec_2[2] = Vec_2[2]/Norm;
        
        
        Vec_3[0] = Vec_2[2]*Vec_1[1]-Vec_2[1]*Vec_1[2];
        Vec_3[1] = Vec_2[0]*Vec_1[2]-Vec_2[2]*Vec_1[0];
        Vec_3[2] = Vec_2[1]*Vec_1[0]-Vec_2[0]*Vec_1[1];
        Norm = sqrtf(Vec_3[0]*Vec_3[0]+Vec_3[1]*Vec_3[1]+Vec_3[2]*Vec_3[2]);
        Vec_3[0] = Vec_3[0]/Norm;
        Vec_3[1] = Vec_3[1]/Norm;
        Vec_3[2] = Vec_3[2]/Norm;
    }

     else if(Vec_1[1] == 0)
     {
        Var_proj = Vec_1[1];
        Vec_2[1] = 1-Var_proj*Vec_1[1];
        Vec_2[0] = -Var_proj*Vec_1[0];
        Vec_2[2] = -Var_proj*Vec_1[2];
        Norm = sqrtf(Vec_2[0]*Vec_2[0]+Vec_2[1]*Vec_2[1]+Vec_2[2]*Vec_2[2]);
        Vec_2[0] = Vec_2[0]/Norm;
        Vec_2[1] = Vec_2[1]/Norm;
        Vec_2[2] = Vec_2[2]/Norm;
        
        
        Vec_3[0] = Vec_2[2]*Vec_1[1]-Vec_2[1]*Vec_1[2];
        Vec_3[1] = Vec_2[0]*Vec_1[2]-Vec_2[2]*Vec_1[0];
        Vec_3[2] = Vec_2[1]*Vec_1[0]-Vec_2[0]*Vec_1[1];
        Norm = sqrtf(Vec_3[0]*Vec_3[0]+Vec_3[1]*Vec_3[1]+Vec_3[2]*Vec_3[2]);
        Vec_3[0] = Vec_3[0]/Norm;
        Vec_3[1] = Vec_3[1]/Norm;
        Vec_3[2] = Vec_3[2]/Norm;
    }
    else if(Vec_1[2] == 0)
     {
        Var_proj = Vec_1[2];
        Vec_2[2] = 1-Var_proj*Vec_1[2];
        Vec_2[1] = -Var_proj*Vec_1[1];
        Vec_2[0] = -Var_proj*Vec_1[0];
        Norm = sqrtf(Vec_2[0]*Vec_2[0]+Vec_2[1]*Vec_2[1]+Vec_2[2]*Vec_2[2]);
        Vec_2[0] = Vec_2[0]/Norm;
        Vec_2[1] = Vec_2[1]/Norm;
        Vec_2[2] = Vec_2[2]/Norm;
        
        
        Vec_3[0] = Vec_2[2]*Vec_1[1]-Vec_2[1]*Vec_1[2];
        Vec_3[1] = Vec_2[0]*Vec_1[2]-Vec_2[2]*Vec_1[0];
        Vec_3[2] = Vec_2[1]*Vec_1[0]-Vec_2[0]*Vec_1[1];
        Norm = sqrtf(Vec_3[0]*Vec_3[0]+Vec_3[1]*Vec_3[1]+Vec_3[2]*Vec_3[2]);
        Vec_3[0] = Vec_3[0]/Norm;
        Vec_3[1] = Vec_3[1]/Norm;
        Vec_3[2] = Vec_3[2]/Norm;
    }

    // Copy Result into matrix

    Input_Mat[1] = Vec_2[0];
    Input_Mat[4] = Vec_2[1];
    Input_Mat[7] = Vec_2[2];
    Input_Mat[2] = Vec_3[0];
    Input_Mat[5] = Vec_3[1];
    Input_Mat[8] = Vec_3[2];

}

__device__ void Orth_Plane(float* direction,float alpha,float* orth_vec_new)
{

    float Q[9];

    for(int i = 0; i < 9;++i)
    {
        Q[i] = 0;
    }

    for(int j = 0; j< 3;++j)
    {
        Q[j*3] = direction[j];
    }

    Norm_Null_Space(&Q[0]);
    
    for(int k = 0; k < 3;++k)
    {
        orth_vec_new[k] = cosf(alpha)*Q[k*3+1]+sinf(alpha)*Q[k*3+2];
    }
}


__device__ void Get_Mean_Diff_Disk(float* Data, float radius, float* Input_or, float* orientations, int X_dim, int*\
                        Node_Coord, float * Func_mean_disk, int num_orientations,float Rad_min,int Rad_num, int Num_Theta)
{

    float Orth_Vec[3] = {0,0,0};

    int Node_Coord_1[3] = {0,0,0};

    float Theta_Angle = 0;

    float Radius = 0;

    int Str_X = (int) X_dim;

    int Str_or = (int) num_orientations;

    for(int i=0; i < int(num_orientations);++i)
    {
        for(int j=0; j < Rad_num;++j)
        {
            for(int k=0; k < Num_Theta;++k)
            {
                Theta_Angle = k*(2*CUDART_PI_F/(Num_Theta-1));

                Radius = Rad_min+(j*radius)/(Rad_num-1);

                Orth_Plane(Input_or, Theta_Angle, &Orth_Vec[0]);
            
                Spatial_Coordinates_Disk(&Node_Coord[0], &Node_Coord_1[0] ,&Radius, &Theta_Angle,&Orth_Vec[0],X_dim);

                Func_mean_disk[i] += 1.0/(radius*radius)*Data[Node_Coord_1[0]*Str_X*Str_X*Str_or+Node_Coord_1[1]*Str_X*Str_or+\
                     Node_Coord_1[2]*Str_or+i]*Radius*(1.0/Rad_num)*(2*CUDART_PI_F/Num_Theta);
            }
        }

    }

}


__device__ void Get_Mean_Diff_Annular(float* Data, float radius, float* Input_or, float* orientations, int X_dim, int*\
                        Node_Coord, float * Func_mean_annular, int num_orientations,float alpha,float Rad_min,int Rad_num, int Num_Theta)
{
    float Orth_Vec[3] = {0,0,0};

    int Node_Coord_1[3] = {0,0,0};

    float Theta_Angle = 0;

    float Radius = 0;
    int Str_X = (int) X_dim;

    int Str_or = (int) num_orientations;

    for(int i=0; i < int(num_orientations);++i)
    {
        for(int j=0; j < Rad_num;++j)
        {
            for(int k=0; k < Num_Theta;++k)
            {
                Theta_Angle = k*(2*CUDART_PI_F/(Num_Theta-1));

                Radius = Rad_min+radius+(j*(radius*alpha-radius))/(Rad_num-1);
                
                //printf("Radius",Radius);
                
                Orth_Plane(Input_or, Theta_Angle, &Orth_Vec[0]);
            
                Spatial_Coordinates_Disk(&Node_Coord[0], &Node_Coord_1[0] ,&Radius, &Theta_Angle,&Orth_Vec[0],X_dim);

                Func_mean_annular[i] += 1.0/((alpha*alpha-1)*radius*radius)*Data[Node_Coord_1[0]*Str_X*Str_X*Str_or+Node_Coord_1[1]*Str_X*Str_or+\
                     Node_Coord_1[2]*Str_or+i]*Radius*(1.0/Rad_num)*(2*CUDART_PI_F/Num_Theta);
            }
        }

    }

}



extern "C" __global__
void energy_radius(float* Data, float* radii_gpu, float* orientations_gpu,\
     float* Potential_Ret,long long X_dim, long long Y_dim, long long Z_dim,\
                             long long num_orientations, long long num_radii,float Rad_min,long long Rad_num,long long Num_Theta)
{

    const int block_ix = blockIdx.x;
    const int block_iy = blockIdx.y;
    const int block_iz = blockIdx.z;

    const int block_sx = blockDim.x;
    const int block_sy = blockDim.y;
    const int block_sz = blockDim.z;

    const int thread_ix = threadIdx.x;
    const int thread_iy = threadIdx.y;
    const int thread_iz = threadIdx.z;

 

    const int k = block_ix*block_sx + thread_ix;
    const int l = block_iy*block_sy + thread_iy;
    const int i = block_iz*block_sz + thread_iz;



    if(k >= (int) X_dim) {return;}
    if(l >= (int) Y_dim) {return;}
    if(i >= (int) Z_dim) {return;}

    int Node_Coord[3] = {k,l,i};

    int Pos_Index =  0;

    float Func_mean_disk[68];

    float Func_mean_disk_annular[68];

    for(int k = 0;k<68;++k)
    {
        Func_mean_disk_annular[k] = 0;
        Func_mean_disk[k] = 0;
    }

    float alpha = 1.5;

    for(int j=0;j < int(num_orientations);++j)
    {
        for(int s=0; s < int(num_radii);++s)
        {
            Get_Mean_Diff_Disk( &Data[0],radii_gpu[s],&orientations_gpu[j*3], &orientations_gpu[0],\
                                            int(X_dim), &Node_Coord[0], &Func_mean_disk[0],int(num_orientations),Rad_min,int(Rad_num),int(Num_Theta));
            Get_Mean_Diff_Annular(&Data[0],radii_gpu[s],&orientations_gpu[j*3], &orientations_gpu[0],\
                                             int(X_dim), &Node_Coord[0], &Func_mean_disk_annular[0],int(num_orientations),alpha,Rad_min,int(Rad_num),int(Num_Theta));
            
            Pos_Index = get_index_gpu(k,l,i,j,s,int(X_dim),int(Y_dim),int(Z_dim),int(num_orientations),int(num_radii));

            Potential_Ret[Pos_Index] = 1.0/(1+Spherical_Norm(&Func_mean_disk[0],&Func_mean_disk_annular[0],num_orientations,\
                                                            &orientations_gpu[0]));
        }
    }
}
