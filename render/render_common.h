#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <helper_cuda.h>

#include <vector>
#include <string>
#include <unordered_map>
#include <utility>

#define BATCH_NUM(THREADS) ((THREADS) + THREADS_PER_BATCH - 1) / THREADS_PER_BATCH

#define MATH_PI 3.1415927f
#define MAX_VIEW_CNT 10
#define MAX_RENDER_NUM 4
#define GRAD_STEP 0.001f
#define GRAD_STEP_NORMAL 0.001f

#define GLM_FORCE_CUDA
#define GLM_FORCE_SWIZZLE
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#define SPE_LOD_OFFSET 1.f
#define DIFF_LOD_OFFSET 2.f

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

//typedef texture<float4, cudaTextureTypeCubemap> textureCube;

// extern std::unordered_map<std::string, const float*>                            matWorldMap;
// extern std::unordered_map<std::string, const float*>                            worldPosMap;
// extern std::unordered_map<std::string, const float*>                            eyePosWorldMap;

// extern std::vector<cudaArray*>                                                  lightMapList;
// extern std::vector<cudaTextureObject_t>                                         texObjectList;

std::vector<cudaMipmappedArray_t>                           lightMapList[MAX_RENDER_NUM];
std::vector<cudaTextureObject_t>                            texObjectList[MAX_RENDER_NUM];                                   
cudaTextureObject_t*                                        texObjectList_GPU[MAX_RENDER_NUM];


cudaArray_t                                                 lightMap_AreaLight[MAX_RENDER_NUM];
cudaTextureObject_t                                         texObjectList_AreaLight[MAX_RENDER_NUM];                                                   
std::unordered_map<std::string, int>                        renderInstanceMap;

//bool                                                                        light_init_tag = false;

// std::unordered_map<std::string, float*>                                     matWorldMap;
// std::unordered_map<std::string, float*>                                     matViewMap;
// std::unordered_map<std::string, float*>                                     eyePosWorldMap;


__device__ float radicalInverse_VdC(uint bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

__device__ glm::vec2 Hammersley(uint i, uint NumSamples)
{
    return glm::vec2(float(i) / float(NumSamples), radicalInverse_VdC(i));
}


// PDFs
// Beckmann
// http://blog.selfshadow.com/publications/s2012-shading-course/
// PDF_L = PDF_H /(4 * VoH)
__device__ float PDF_Beckmann_H(float roughness, float NoH)
{
    float e = exp((NoH * NoH - 1) / (NoH * NoH * roughness * roughness));

    // beckmann * NoH
    return e / (MATH_PI * roughness * roughness * NoH * NoH * NoH);
}

// Diffuse
__device__ float PDF_Diffuse(float NoL)
{
    return NoL / MATH_PI;
}

//Geometry Term
__device__ float G_SchlickSmith(float roughness, float NoL, float NoV)
{
    float a = roughness * sqrt(2.f / MATH_PI);
    float visInv = (NoL * (1.f - a) + a) * (NoV *(1.f - a) + a);
    return NoL * NoV / visInv;
}

//Cook-Torrance Geometric term
__device__ float G_CookTorrance(float NoL, float NoV, float NoH, float VoH)
{
    float shad1 = (2.0f * NoH * NoV) / (VoH);
    float shad2 = (2.0f * NoH * NoL) / (VoH);
    return min(1.0f, min(shad1, shad2));
}


// Importance Sampling Functions
// Diffuse
__device__ glm::vec3 ImportanceSampleDiffuse(const glm::vec2& Xi, const glm::vec3& N)
{
    float Phi = 2 * MATH_PI * Xi.x;
    float CosTheta = sqrt(1 - Xi.y);
    float SinTheta = sqrt(1 - CosTheta * CosTheta);
    glm::vec3 H;
    H.x = SinTheta * cos(Phi);
    H.y = SinTheta * sin(Phi);
    H.z = CosTheta;

    glm::vec3 UpVector = abs(N.z) < 0.5f ? glm::vec3(0, 0, 1) : glm::vec3(1, 0, 0);
    //float3 UpVector = normalize((1 - abs(N.z + N.x + N.y)) * float3(0, 0, 1) + 0.5f * abs(N.z + N.x + N.y) * float3(1, 0, 0));
    glm::vec3 TangentX = glm::normalize(glm::cross(UpVector, N));
    glm::vec3 TangentY = glm::normalize(glm::cross(N, TangentX));
    // Tangent to world space
    return TangentX * H.x + TangentY * H.y + N * H.z;
}


// Beckmann
// http://www.cs.cornell.edu/~srm/publications/egsr07-btdf.pdf
__device__ glm::vec3 ImportanceSampleBeckmann(const glm::vec2& Xi, float roughness, const glm::vec3& N)
{
    float Phi = 2 * MATH_PI * Xi.x;
    float CosTheta = sqrt(1.f / (1 - roughness * roughness * log(1 - Xi.y)));
    float SinTheta = sqrt(1 - CosTheta * CosTheta);
    glm::vec3 H;
    H.x = SinTheta * cos(Phi);
    H.y = SinTheta * sin(Phi);
    H.z = CosTheta;
    glm::vec3 UpVector = abs(N.z) < 0.5f ? glm::vec3(0, 0, 1) : glm::vec3(1, 0, 0);
    //float3 UpVector = normalize((1 - abs(N.z + N.x + N.y)) * float3(0, 0, 1) + 0.5f * abs(N.z + N.x + N.y) * float3(1, 0, 0));
    glm::vec3 TangentX = glm::normalize(glm::cross(UpVector, N));
    glm::vec3 TangentY = glm::normalize(glm::cross(N, TangentX));
    // Tangent to world space
    return TangentX * H.x + TangentY * H.y + N * H.z;
}


__device__ glm::vec3 vectorTransform(const glm::mat4& mat, const glm::vec3& vector)
{
    glm::vec4 v4(vector, 0.0f);
    glm::vec4 out4 = mat * v4;
    return glm::vec3(out4.x, out4.y, out4.z);
}

__device__ glm::vec3 xyToWorldSpaceNrm(float x, float y, const glm::mat4& matWorld)
{
    float z = sqrt(1.0 - x*x - y*y);
    glm::vec4 oNrm(x, y, z, 0.0f);
    glm::vec4 nrm_xform = matWorld * oNrm;
    return glm::vec3(nrm_xform.x, nrm_xform.y, nrm_xform.z);
}

__device__ glm::vec3 xyToLightSpaceNrm(float x, float y, const glm::mat4& matWorld, const glm::mat4& matLight)
{
    float z = sqrt(1.0 - x*x - y*y);
    glm::vec4 oNrm(x, y, z, 0.0f);
    glm::vec4 nrm_xform = matLight * matWorld * oNrm;
    return glm::vec3(nrm_xform.x, nrm_xform.y, nrm_xform.z);
}

__device__ glm::vec3 thetaPhiToWorldSpaceNrm(float theta, float phi, const glm::mat4& matWorld)
{
    glm::vec4 oNrm(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta), 0.0f);
    glm::vec4 nrm_xform = matWorld * oNrm;
    return glm::vec3(nrm_xform.x, nrm_xform.y, nrm_xform.z);
}

__device__ glm::vec3 thetaPhiToLightSpaceNrm(float theta, float phi, const glm::mat4& matWorld, const glm::mat4& matLight)
{
    glm::vec4 oNrm(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta), 0.0f);
    glm::vec4 nrm_xform = matLight * matWorld * oNrm;
    return glm::vec3(nrm_xform.x, nrm_xform.y, nrm_xform.z);
}


__device__ glm::vec3 texCubeSampler(const cudaTextureObject_t tex, const glm::vec3& vec, float lod = 0.0f)
{
    float4 color_4 = texCubemapLod<float4>(tex, vec.x, vec.y, vec.z, lod);
    return glm::vec3(color_4.x, color_4.y, color_4.z);
}

__device__ glm::vec3 EvalDiffuseShadingForSingleRay(const glm::vec3& N, const glm::vec3& L, cudaTextureObject_t tex, const float solidAng_pixel, const int nSamples)
{
    glm::vec3 output(0.0f);

    float NoL = glm::clamp(glm::dot(N, L), 0.0f, 1.0f);
    if(NoL > 1e-6)
    {
        float solidAng_sample = 1.f / (nSamples * PDF_Diffuse(NoL));
        float lod = min(8.0f, max(0.0f, 0.5f * log2(solidAng_sample / solidAng_pixel)) + DIFF_LOD_OFFSET);
        output = texCubeSampler(tex, L, lod);
    }
    if(isnan(output.x) || isnan(output.y) || isnan(output.z))
    {
        output = glm::vec3(0.0f);
    }
    return output;
}


__device__ glm::vec3 EvalSpecShadingForSingleRay(const glm::vec3& N, const glm::vec3& V, const glm::vec3& L, const glm::vec3& H,
                                                 const glm::vec3& spec, float roughness,
                                                 cudaTextureObject_t tex, const float solidAng_pixel, const int nSamples, const bool useFresnel)
{
    glm::vec3 output(0.0f);

    float NoV = glm::clamp(glm::dot(N, V), 0.0f, 1.0f) + 1e-6f;
    float NoL = glm::clamp(glm::dot(N, L), 0.0f, 1.0f);
    float NoH = glm::clamp(glm::dot(N, H), 0.0f, 1.0f) + 1e-6f;
    float VoH = glm::clamp(glm::dot(V, H), 0.0f, 1.0f) + 1e-6f;

    if(NoL > 1e-6)
    {
        float G = G_CookTorrance(NoL, NoV, NoH, VoH);
        float solidAng_sample = 4.f * VoH / (nSamples * PDF_Beckmann_H(roughness, NoH));
        float lod = min(8.0f, max(0.0f, 0.5f * log2(solidAng_sample / solidAng_pixel)) + SPE_LOD_OFFSET);

        glm::vec3 F(0.0f);
        float fg = 1.0f;

        if(useFresnel)
        {
            float Fc = pow(1.0f - VoH, 5);
            fg = 1.0f - Fc;
            F = fg * spec + Fc;
        }
        else
            F = spec;

        glm::vec3 sampledColor = texCubeSampler(tex, L, lod);
        glm::vec3 tmp = sampledColor * G * VoH / (NoH * NoV);
        output = tmp * F;
        if(isnan(output.x) || isnan(output.y) || isnan(output.z))
        {
           output = glm::vec3(0.0f);
        }
    }

    return output;
}



__device__ glm::vec3 EvalDiffuseShadingForPointLight(const glm::vec3& L, const glm::vec3& N)
{
    float d = glm::clamp(glm::dot(L, N), 0.0f, 1.0f) / MATH_PI;
    return glm::vec3(d, d, d);
}

__device__ glm::vec3 EvalSpecShadingForPointLight(const glm::vec3& N, const glm::vec3& V, const glm::vec3& L, const glm::vec3& H,
                                                  const glm::vec3& spec, float roughness, bool useFresnel)
{
    glm::vec3 output(0.0f);

    float NoV = glm::clamp(glm::dot(N, V), 0.0f, 1.0f) + 1e-10f;
    float NoL = glm::clamp(glm::dot(N, L), 0.0f, 1.0f);
    float NoH = glm::clamp(glm::dot(N, H), 0.0f, 1.0f) + 1e-10f;
    float VoH = glm::clamp(glm::dot(V, H), 0.0f, 1.0f) + 1e-10f;

    float D = PDF_Beckmann_H(roughness, NoH);
    if(NoL > 0)
    {
        float G = G_CookTorrance(NoL, NoV, NoH, VoH);
        glm::vec3 F(0.0f);

        if(useFresnel)
        {
            float Fc = pow(1.0f - VoH, 5);
            F = (1.0f - Fc) * spec + Fc;
        }
        else
            F = spec;

        output = D * G * F / (4.0f * NoV);
    }

    return output;
}



__device__ bool intersectPlane(const glm::vec3& viewDir, const glm::vec3& viewOrigin,
                               const glm::vec3& planeOrigin, const glm::vec3& planeNormal,
                               glm::vec3& output)
{
    output = glm::vec3(0.0f);
    if(glm::dot(viewDir, planeNormal) == 0)
        return false;

    float t =  glm::dot(planeNormal, planeOrigin - viewOrigin) / glm::dot(planeNormal, viewDir);
    if(t <= 0)
        return false;
    
    output = viewOrigin + t * viewDir;    
    return true;
}

__device__ bool intersectSphere(const glm::vec3& viewDir, const glm::vec3& viewOrigin,
                               const glm::vec3& sphereOrigin, const float radius,
                               glm::vec3& output)
{
    output = glm::vec3(0.0f);
    float dotProd = glm::dot(viewDir, viewOrigin - sphereOrigin);
    float sqr_distOrigin = glm::dot(viewOrigin - sphereOrigin, viewOrigin - sphereOrigin);

    float tag = dotProd * dotProd - sqr_distOrigin + radius * radius;
    if(tag < 0)
        return false;
    
    float t = -dotProd - sqrt(tag);
    output = viewOrigin + t * viewDir;
    return true;
}

__device__ bool isPixelVaild(const glm::vec3& viewDir, const glm::vec3& viewOrigin,
                             const glm::vec3& planeOrigin, 
                             const glm::vec3& planeX, const glm::vec3& planeY, const glm::vec3& planeNormal,
                             const float planeWidth, const float planeHeight)
{
    glm::vec3 pointOnPlane;
    if(intersectPlane(viewDir, viewOrigin, planeOrigin, planeNormal, pointOnPlane) == false)
        return false;

    float x = glm::dot(pointOnPlane - planeOrigin, planeX);
    float y = glm::dot(pointOnPlane - planeOrigin, planeY);

    if(x <= planeWidth / 2 && x >= -planeWidth / 2 && y <= planeHeight / 2 && y >= -planeHeight / 2)
        return true;
    else
        return false;
}



void reportGPUMemory()
{
    size_t free_byte ;
    size_t total_byte ;
    checkCudaErrors(cudaMemGetInfo(&free_byte, &total_byte));
    double free_db = (double)free_byte;
    double total_db = (double)total_byte;
    double used_db = total_db - free_db;

    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
    used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);    
}