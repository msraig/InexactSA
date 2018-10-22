#include "render_common.h"
#include "imageIO.h"
#include <fstream>
#define PLANE_WIDTH 2.0f
#define BALL_RADIUS 1.0f
#define MAX_THREADS_PER_BLOCK 256
#define THREADS_PER_BATCH 256

//0: image 1: albedo 2: spec 3: roughness 4: normal

__global__ void init_camera(float* out_view_dir,
                            const float* eyePosWorld_data, 
                            const float* matView_data, const float* matProj_data,
                            const int imgHeight, const int imgWidth,
                            const uint threads)
{
    uint index = threadIdx.x + blockDim.x * blockIdx.x;
    if(index >= threads)
        return;    

    int point_x = index % imgWidth;
    int point_y = imgHeight - index / imgWidth;

    //glm matrices
    glm::mat4 matView           = glm::transpose(glm::make_mat4(matView_data));
    glm::mat4 matProj           = glm::transpose(glm::make_mat4(matProj_data));

    //glm::mat4 matWorldView = matView * matWorld;

    //get vaild view direction
    // glm::vec3 planeOrigin(matWorld[3].x, matWorld[3].y, matWorld[3].z);
    // glm::vec3 planeX(matWorld[0].x, matWorld[0].y, matWorld[0].z); 
    // glm::vec3 planeY(matWorld[1].x, matWorld[1].y, matWorld[1].z);
    // glm::vec3 planeNormal(matWorld[2].x, matWorld[2].y, matWorld[2].z);

    glm::vec3 wEyePos(eyePosWorld_data[0], eyePosWorld_data[1], eyePosWorld_data[2]);
    glm::vec3 screenPos(point_x, point_y, 1.0f);
    glm::vec3 screenPosInWorldSpace = glm::unProject(screenPos, matView, matProj, glm::vec4(0, 0, imgWidth, imgHeight));

    glm::vec3 viewDir = glm::normalize(screenPosInWorldSpace - wEyePos);


    out_view_dir[3*index] = -viewDir.x;
    out_view_dir[3*index+1] = -viewDir.y;
    out_view_dir[3*index+2] = -viewDir.z;
}




__global__ void render(float* output_diffuse_sum, float* output_spec_sum,  
                       const float* albedo_data, const float* spec_data, const float* roughness_data, const float* normal_data,
                       const float* mask_data,
                       const float* matWorld_data, const float* matWorld2Light_data, const float* viewDir_data,
                       const int* light_id_data, const cudaTextureObject_t* texObjectList_GPU_tmp,
                       const int imgHeight, const int imgWidth,
                       const int nSampleDiffuse, const int nSampleSpec, const int nCubeRes, const bool useFresnel, const int renderMode,
                       const uint threads)
{
    uint index = threadIdx.x + blockDim.x * blockIdx.x;
    if(index >= threads)
        return;

    if(mask_data[index] < 0.5)
    {
        output_diffuse_sum[3*index] = 0.0f;
        output_diffuse_sum[3*index+1] = 0.0f;
        output_diffuse_sum[3*index+2] = 0.0f;

        output_spec_sum[3*index] = 0.0f;
        output_spec_sum[3*index+1] = 0.0f;
        output_spec_sum[3*index+2] = 0.0f;      
        return;                
    }

    int posInBatch = index / (imgWidth * imgWidth);
    int index_img = index % (imgWidth * imgWidth);

    int point_x = index_img % imgWidth;
    int point_y = imgHeight - index_img / imgWidth;

    cudaTextureObject_t tex = texObjectList_GPU_tmp[light_id_data[posInBatch]];

    //glm matrices
    glm::mat4 matWorld          = glm::transpose(glm::make_mat4(&(matWorld_data[16*posInBatch])));
    glm::mat4 matWorld2Light    = glm::transpose(glm::make_mat4(&(matWorld2Light_data[16*posInBatch])));

    //albedo, spec, and roughness
    glm::vec3 albedo(albedo_data[3*index], albedo_data[3*index+1], albedo_data[3*index+2]);
    glm::vec3 spec(spec_data[3*index], spec_data[3*index+1], spec_data[3*index+2]);

    float roughness = roughness_data[index];
    roughness = fmaxf(fminf(roughness, 0.95f), 0.002f);

    if(roughness < 0.0f)
    {
        output_diffuse_sum[3*index] = 0.0f;
        output_diffuse_sum[3*index+1] = 0.0f;
        output_diffuse_sum[3*index+2] = 0.0f;

        output_spec_sum[3*index] = 0.0f;
        output_spec_sum[3*index+1] = 0.0f;
        output_spec_sum[3*index+2] = 0.0f;       

        return;        
    }

//    roughness = fmaxf(fminf(roughness, 0.85f), 0.04f);

    float nrm_x = normal_data[2*index];
    float nrm_y = normal_data[2*index+1];

    glm::vec3 N = xyToLightSpaceNrm(nrm_x, nrm_y, matWorld, matWorld2Light);
    glm::vec3 V(viewDir_data[3*index_img], viewDir_data[3*index_img+1], viewDir_data[3*index_img+2]);
    V = vectorTransform(matWorld2Light, V);


    if(renderMode == 1)
    {
        output_diffuse_sum[3*index] = albedo.x;
        output_diffuse_sum[3*index+1] = albedo.y;
        output_diffuse_sum[3*index+2] = albedo.z;

        output_spec_sum[3*index] = 0.0f;
        output_spec_sum[3*index+1] = 0.0f;
        output_spec_sum[3*index+2] = 0.0f;       
        return;
    }

    if(renderMode == 2)
    {
        output_spec_sum[3*index] = spec.x;
        output_spec_sum[3*index + 1] = spec.y;
        output_spec_sum[3*index + 2] = spec.z;

        output_diffuse_sum[3*index] = 0.0f;
        output_diffuse_sum[3*index+1] = 0.0f;
        output_diffuse_sum[3*index+2] = 0.0f;        
        return;
    }

    if(renderMode == 3)
    {
        output_diffuse_sum[3*index] = roughness;
        output_diffuse_sum[3*index+1] = roughness;
        output_diffuse_sum[3*index+2] = roughness;

        output_spec_sum[3*index] = 0.0f;
        output_spec_sum[3*index+1] = 0.0f;
        output_spec_sum[3*index+2] = 0.0f;           
        return;
    }

    if(renderMode == 4)
    {
        output_diffuse_sum[3*index] = nrm_x;
        output_diffuse_sum[3*index+1] = nrm_y;
        output_diffuse_sum[3*index+2] = sqrt(1.0 - nrm_x*nrm_x - nrm_y*nrm_y);

        output_spec_sum[3*index] = 0.0f;
        output_spec_sum[3*index+1] = 0.0f;
        output_spec_sum[3*index+2] = 0.0f;  
        return;
    }

    //shading accumator
    glm::vec3 diffuseShading(0, 0, 0);
    glm::vec3 specShading(0, 0, 0);

    //for LOD compute
    const float solidAng_pixel = 4 * MATH_PI / (nCubeRes * nCubeRes * 6);

    //compute diffuse shading

    for (int i=0; i<nSampleDiffuse; i++)
    {
        glm::vec2 Xi = Hammersley(i, nSampleDiffuse);
        glm::vec3 L = ImportanceSampleDiffuse(Xi, N);

        diffuseShading += EvalDiffuseShadingForSingleRay(N, L, tex, solidAng_pixel, nSampleDiffuse);
    }


    //multiply albedo
    diffuseShading = diffuseShading * albedo;
    output_diffuse_sum[3*index] = diffuseShading.x;
    output_diffuse_sum[3*index + 1] = diffuseShading.y;
    output_diffuse_sum[3*index + 2] = diffuseShading.z;

    //compute specular shading
    for (int i=0; i<nSampleSpec; i++)
    {
        glm::vec2 Xi = Hammersley(i, nSampleSpec);
        glm::vec3 H = ImportanceSampleBeckmann(Xi, roughness, N);
        glm::vec3 L = 2 * glm::dot(V, H) * H - V;

        specShading += EvalSpecShadingForSingleRay(N, V, L, H, spec, roughness, tex, solidAng_pixel, nSampleSpec, useFresnel);
    }



    output_spec_sum[3*index] = specShading.x;
    output_spec_sum[3*index + 1] = specShading.y;
    output_spec_sum[3*index + 2] = specShading.z;

}


//d*h*w*ch
__device__ glm::vec4 getPixel(const float* lightMap, int x, int y, int d, int width)
{
    int strideD = width*width*4;
    int strideY = width*4;

    int begin = d*strideD + y*strideY + x*4;

    float b = lightMap[begin];
    float g = lightMap[begin+1];
    float r = lightMap[begin+2];
    float a = lightMap[begin+3];

    return glm::vec4(b,g,r,a);
}

__device__ void setPixel(float* lightMap, int x, int y, int d, int width, const glm::vec4& value)
{
    int strideD = width*width*4;
    int strideY = width*4;

    int begin = d*strideD + y*strideY + x*4;
    
    lightMap[begin] = value.x;
    lightMap[begin+1] = value.y;
    lightMap[begin+2] = value.z;
    lightMap[begin+3] = value.w;
    
}

__global__ void downsample_test(const float* lightMap_prev, float* lightMap_curr, int currWidth)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int nThread = currWidth * currWidth * 6 * 4;
    if(index >= nThread)
        return;

    lightMap_curr[index] = 1.0f;
}

__global__ void downsample(const float* lightMap_prev, float* lightMap_curr, int currWidth)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int nThread = currWidth * currWidth * 6;
    if(index >= nThread)
        return;

    int depth = index / (currWidth * currWidth);
    int ypos = (index - depth * currWidth * currWidth) / (currWidth);
    int xpos = index - depth * currWidth * currWidth - ypos * currWidth;

    int prevWidth = currWidth * 2;

    glm::vec4 pixValue = getPixel(lightMap_prev, 2*xpos, 2*ypos, depth, prevWidth) +
                         getPixel(lightMap_prev, 2*xpos+1, 2*ypos, depth, prevWidth) +
                         getPixel(lightMap_prev, 2*xpos, 2*ypos+1, depth, prevWidth) +
                         getPixel(lightMap_prev, 2*xpos+1, 2*ypos+1, depth, prevWidth);
    pixValue /= 4.0f;

    setPixel(lightMap_curr, xpos, ypos, depth, currWidth, pixValue);
}

float* cameraSetup(const float* eyePosWorld_data, 
                 const float* matView_data, const float* matProj_data,
                 const int imgHeight, const int imgWidth)
{

    float* out_view_dir_data = NULL;
    checkCudaErrors(cudaMalloc(&out_view_dir_data, 3*sizeof(float)*imgHeight*imgWidth));


    int nThread = imgHeight * imgWidth;
    
    init_camera<<<BATCH_NUM(nThread), THREADS_PER_BATCH>>>(out_view_dir_data, 
        eyePosWorld_data, 
        matView_data, 
        matProj_data,
        imgHeight,
        imgWidth,
        nThread);

// //    debug
//     float* tmp = (float*)malloc(3*sizeof(float)*imgHeight*imgWidth);
//     cudaMemcpy(tmp, out_view_dir_data, 3*sizeof(float)*imgHeight*imgWidth, cudaMemcpyDeviceToHost);

//     for(int i=0; i<10; i++)
//         printf("%f\n", tmp[i]);

    return out_view_dir_data;

}

void copyTexObjectToGPU(const int render_id)
{
    cudaTextureObject_t* texObjectList_CPU = (cudaTextureObject_t*)malloc(texObjectList[render_id].size() * sizeof(cudaTextureObject_t));
    for(int i=0; i<texObjectList[render_id].size(); i++)
        texObjectList_CPU[i] = texObjectList[render_id][i];

    checkCudaErrors(cudaMalloc(&texObjectList_GPU[render_id], texObjectList[render_id].size() * sizeof(cudaTextureObject_t)));
    cudaMemcpy(texObjectList_GPU[render_id], texObjectList_CPU, sizeof(cudaTextureObject_t) * texObjectList[render_id].size(), cudaMemcpyHostToDevice);

    free(texObjectList_CPU);
}


void renderImage(const std::string& instanceID,
    float* output_diffuse_sum,
    float* output_spec_sum,
    const float* albedo_data,
    const float* spec_data,
    const float* roughness_data,
    const float* normal_data,
    const float* mask_data,
    const float* matWorld2Light_data,    
    const int* lightid_data,
    const float* matWorld_data,
    const float* matView_data,
    const float* matProj_data,
    const float* eyePosWorld_data,
    const int imgHeight, const int imgWidth, const int batchSize,
    const int nSampleDiffuse, const int nSampleSpec, const int nCubeRes,
    const bool useFresnel, const int renderMode)
{
    cudaMemset(output_diffuse_sum, 0, batchSize * sizeof(float) * 3 * imgHeight * imgWidth);
    cudaMemset(output_spec_sum, 0, batchSize * sizeof(float) * 3 * imgHeight * imgWidth);

    int threads = imgHeight * imgWidth * batchSize;        
    float* viewDir_data = cameraSetup(eyePosWorld_data, matView_data, matProj_data, imgHeight, imgWidth);

    int render_id = renderInstanceMap[instanceID];

    render<<<BATCH_NUM(threads), THREADS_PER_BATCH>>>(
        output_diffuse_sum, output_spec_sum, 
        albedo_data, spec_data, roughness_data, normal_data, 
        mask_data,
        matWorld_data, matWorld2Light_data, 
        viewDir_data, lightid_data, texObjectList_GPU[render_id],
        imgHeight, imgWidth, 
        nSampleDiffuse, nSampleSpec, nCubeRes, useFresnel, renderMode, threads);

    checkCudaErrors(cudaFree(viewDir_data));

}


cudaMipmappedArray_t generateMipMaps(float* lightMap_current, const int envWidth)
{
    //calc mip level
    int nLevel = int(log2(float(envWidth)) + 1);

    //init mipmap array
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
    cudaMipmappedArray_t mipmapArray;
    checkCudaErrors(cudaMallocMipmappedArray(&mipmapArray, &desc, make_cudaExtent(envWidth, envWidth, 6), nLevel, cudaArrayCubemap));

    // upload level 0
    cudaArray_t level0;
    checkCudaErrors(cudaGetMipmappedArrayLevel(&level0, mipmapArray, 0));
    cudaMemcpy3DParms copyparams = {0};
    copyparams.srcPos = make_cudaPos(0,0,0);
    copyparams.dstPos = make_cudaPos(0,0,0);
    copyparams.srcPtr = make_cudaPitchedPtr(lightMap_current, envWidth * sizeof(float4), envWidth, envWidth);
    copyparams.dstArray = level0;
    copyparams.extent = make_cudaExtent(envWidth, envWidth, 6);
    copyparams.kind = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaMemcpy3D(&copyparams));
    
    int currWidth = envWidth / 2;
    float* lightMap_level[nLevel];

    float* lightMap_level_0 = NULL;
    checkCudaErrors(cudaMalloc(&lightMap_level_0, 4*sizeof(float)*6*envWidth*envWidth));
    checkCudaErrors(cudaMemcpy(lightMap_level_0, lightMap_current, 4*sizeof(float)*6*envWidth*envWidth, cudaMemcpyHostToDevice));
    lightMap_level[0] = lightMap_level_0;

    for(int i=1; i<nLevel; i++)
    {
        float* lightMap_level_prev = lightMap_level[i-1];
        float* lightMap_level_curr = NULL;
        checkCudaErrors(cudaMalloc(&lightMap_level_curr, 4*sizeof(float)*6*currWidth*currWidth));

        int nThread = currWidth*currWidth*6;
        downsample<<<BATCH_NUM(nThread), THREADS_PER_BATCH>>>(lightMap_level_prev, lightMap_level_curr, currWidth);
        checkCudaErrors(cudaMemcpy(lightMap_level_0, lightMap_current, 4*sizeof(float)*6*envWidth*envWidth, cudaMemcpyHostToDevice));      
        
        cudaArray_t level_curr;
        checkCudaErrors(cudaGetMipmappedArrayLevel(&level_curr, mipmapArray, i));

        cudaMemcpy3DParms copyparams_curr = {0};
        copyparams_curr.srcPos = make_cudaPos(0,0,0);
        copyparams_curr.dstPos = make_cudaPos(0,0,0);
        copyparams_curr.srcPtr = make_cudaPitchedPtr(lightMap_level_curr, currWidth * sizeof(float4), currWidth, currWidth);
        copyparams_curr.dstArray = level_curr;
        copyparams_curr.extent = make_cudaExtent(currWidth, currWidth, 6);
        copyparams_curr.kind = cudaMemcpyDeviceToDevice;
        checkCudaErrors(cudaMemcpy3D(&copyparams_curr));
    
        currWidth = currWidth / 2;
        lightMap_level[i] = lightMap_level_curr;
    }

    for(int i=0; i<nLevel; i++)
        cudaFree(lightMap_level[i]);

    return mipmapArray;
}


void initLightMap(const std::string& id, const std::string& lightfolder, const std::string& lightfile, const int envWidth)
{
    if(renderInstanceMap.size() >= MAX_RENDER_NUM)
    {
        printf("Max render num exceeded.\n");
        return;        
    }

    if(renderInstanceMap.find(id) != renderInstanceMap.end())
    {
        return;
    }

    int last_render_id = -1;
    printf("Current render list...\n");
    for (auto iter = renderInstanceMap.begin(); iter != renderInstanceMap.end(); ++iter)
    {
        printf("...%s: %d\n", iter->first.c_str(), iter->second);
        if(iter->second > last_render_id)
            last_render_id = iter->second;
    }

    int current_render_id = last_render_id + 1;
    printf("Add a render: id %d\n", current_render_id);
    renderInstanceMap.insert(std::make_pair(id, current_render_id));

    //read light pathes
    std::fstream f_light;
    std::vector<int> light_id_list;
    f_light.open(lightfolder + "/" + lightfile, std::ios::in);
    int tmp_id;
    while(f_light >> tmp_id)
        light_id_list.push_back(tmp_id);
    f_light.close();


    int lightMapCount = light_id_list.size();
    lightMapList[current_render_id].resize(lightMapCount);
    texObjectList[current_render_id].resize(lightMapCount);

    for(int i = 0; i < lightMapCount; i++)
    {
        char tmp_path[20];
        sprintf(tmp_path, "%04d.pfm", light_id_list[i]);
        std::string str_lightfile = tmp_path;
        float* lightMap_current = loadCubeMap(lightfolder + "/" + str_lightfile, envWidth);

        cudaMipmappedArray_t cu_MipArray = generateMipMaps(lightMap_current, envWidth);
        lightMapList[current_render_id][i] = cu_MipArray;

        cudaTextureObject_t texMap;
        
        cudaResourceDesc    resDescr;
        memset(&resDescr,0,sizeof(cudaResourceDesc));

        resDescr.resType = cudaResourceTypeMipmappedArray;
        resDescr.res.mipmap.mipmap = cu_MipArray;
        
        cudaTextureDesc texDescr;
        memset(&texDescr,0,sizeof(cudaTextureDesc));
        texDescr.normalizedCoords = true;
        texDescr.filterMode       = cudaFilterModeLinear;
        texDescr.addressMode[0]   = cudaAddressModeWrap;
        texDescr.addressMode[1]   = cudaAddressModeWrap;
        texDescr.addressMode[2]   = cudaAddressModeWrap;
        texDescr.readMode         = cudaReadModeElementType;
        texDescr.mipmapFilterMode = cudaFilterModeLinear;
        texDescr.minMipmapLevelClamp = 0;
        texDescr.maxMipmapLevelClamp = int(log2(float(envWidth)));

        checkCudaErrors(cudaCreateTextureObject(&texMap, &resDescr, &texDescr, NULL));
        texObjectList[current_render_id][i] = texMap;
        
        free(lightMap_current);
    }

    copyTexObjectToGPU(current_render_id);
}
    
void copyLightIDFromDevicePtr(const int* pDevice, int* pHost, int batchSize)
{
    //Assume host pointer is inited outside!
    cudaMemcpy(pHost, pDevice, sizeof(int) * batchSize, cudaMemcpyDeviceToHost);
}


void freeRenderOp(const std::string& id)
{
    if(renderInstanceMap.find(id) == renderInstanceMap.end())
        return;

    int render_id = renderInstanceMap[id];
    if(texObjectList_GPU[render_id] != NULL)
    {
        checkCudaErrors(cudaFree(texObjectList_GPU[render_id]));
        texObjectList_GPU[render_id] = NULL;        
    }

    for(int i = 0; i < lightMapList[render_id].size(); i++)
    {
        if(texObjectList[render_id][i])
            checkCudaErrors(cudaDestroyTextureObject(texObjectList[render_id][i]));

        if(lightMapList[render_id][i])
            checkCudaErrors(cudaFreeMipmappedArray(lightMapList[render_id][i]));           
    }
    lightMapList[render_id].clear();
    texObjectList[render_id].clear();

    renderInstanceMap.erase(id);
}

